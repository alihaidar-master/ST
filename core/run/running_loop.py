import sys
import time
import torch.distributed
from miscellanies.torch.checkpoint import dump_checkpoint, load_checkpoint
from tqdm import tqdm
import datetime
from contextlib import nullcontext
from core.run.metric_logger.context import enable_logger
from torch import set_grad_enabled
import gc
import cv2 as cv
import numpy as np
import torch
from data.tracking.methods.sequential.curation_parameter_provider import SiamFCCurationParameterSimpleProvider
import torch
from torchvision.transforms.functional import normalize
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from core.run.run_tracking import run_tracking



def print_model_efficiency_assessment(efficiency_assessor, model, wandb_instance):
    flop_count_analysis = efficiency_assessor.get_flop_count_analysis(model)
    print('Initialization modules flop table')
    print(flop_count_analysis.get_flop_count_table_init())

    print('Tracking modules flop table')
    print(flop_count_analysis.get_flop_count_table_track())

    init_fps, track_fps = efficiency_assessor.test_fps(model)

    if wandb_instance is not None:
        wandb_instance.summary.update({'model_mac_init': flop_count_analysis.get_model_mac_init(), 'model_mac_track': flop_count_analysis.get_model_mac_track()})

    print(f"Estimated model FPS: init {init_fps:.3f} track {track_fps:.3f}")


def _wandb_watch_model(model, wandb_instance, watch_model_parameters, watch_model_gradients, watch_model_freq):
    if watch_model_parameters and watch_model_gradients:
        watch_model = 'all'
    elif watch_model_parameters:
        watch_model = 'parameters'
    elif watch_model_gradients:
        watch_model = 'gradients'
    else:
        watch_model = None

    wandb_instance.watch(model, log=watch_model, log_freq=watch_model_freq)


def get_model(model: torch.nn.Module):
    if model.__class__.__name__ == 'DistributedDataParallel':
        return model.module
    else:
        return model
        

def crop_and_resize(img, center, size, out_size,
                    border_type=cv.BORDER_CONSTANT,
                    border_value=(0, 0, 0),
                    interp=cv.INTER_LINEAR):
    # convert box to corners (0-indexed)
    size = round(size)
    corners = np.concatenate((
        np.round(center - (size - 1) / 2),
        np.round(center - (size - 1) / 2) + size))
    corners = np.round(corners).astype(int)

    # pad image if necessary
    pads = np.concatenate((
        -corners[:2], corners[2:] - img.shape[:2]))
    npad = max(0, int(pads.max()))
    if npad > 0:
        img = cv.copyMakeBorder(
            img, npad, npad, npad, npad,
            border_type, value=border_value)

    # crop image patch
    corners = (corners + npad).astype(int)
    patch = img[corners[0]:corners[2], corners[1]:corners[3]]

    # resize to out_size
    patch = cv.resize(patch, (out_size, out_size),
                       interpolation=interp)
    return patch

def numpy_to_torch(a: np.ndarray):
    return torch.from_numpy(a).float().permute(2, 0, 1).unsqueeze(0)

class RunTrackerData:
    def __init__(self,data) -> None:
        self.z_curated = data['z_curated']
        self.z_bbox = data['z_bbox']
        self.z_image_mean = data['z_image_mean']
        self.full_image = data['x']
        self.frame_index = data['frame_index']
        
def run_tracker(model,runner,branch_name):
    device = runner.tracker_evaluator[branch_name].device
    search_image_curation_parameter_provider  = runner.tracker_evaluator[branch_name].search_curation_parameter_provider
    search_curation_image_size = runner.tracker_evaluator[branch_name].search_curation_image_size
    bounding_box_post_processor = runner.tracker_evaluator[branch_name].bounding_box_post_processor
    post_processor = runner.tracker_evaluator[branch_name].post_processor
    interpolation_mode = runner.tracker_evaluator[branch_name].interpolation_mode
    template_curated_image_cache_shape = runner.tracker_evaluator[branch_name].template_curated_image_cache_shape

    is_training = False
    model.train(is_training)
    torch.no_grad()
    if not is_training:
        model = get_model(model)
        model.eval()

    #model is ready here
        video_pth = 'C:/Users/aalih/Documents/SwinTrack/SWMV/test.mp4'
        cap = cv.VideoCapture(video_pth)
        display_name = 'Display: swinTrack'
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        frame_RGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        cv.imshow(display_name, frame)
        while True:
            cv.putText(frame, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                        1.5, (0, 0, 0), 1)
            x, y, w, h = cv.selectROI(display_name, frame, fromCenter=False)
            init_state = [x, y, w, h]
            full_image = numpy_to_torch(frame_RGB).squeeze(0).to(device=device)
            box_corner = [x,y,x+h-1,y+w-1]
            box = init_state
            box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32) # [yc,xc,h,w]
            center,target_sz = box[:2], box[2:]
            # exemplar and search sizes
            cfg_context = 0.5
            cfg_instance_sz = search_curation_image_size[0]
            cfg_exemplar_sz = template_curated_image_cache_shape[3]
            context = cfg_context * np.sum(target_sz)
            z_sz = np.sqrt(np.prod(target_sz + context))
            x_sz = z_sz * \
            cfg_instance_sz / cfg_exemplar_sz
            # exemplar image
            z_curated = crop_and_resize(
                frame_RGB, center, z_sz,
                out_size=cfg_exemplar_sz)
            z_curated_tensor = torch.from_numpy(z_curated).float().permute(2, 0, 1)

            z_curated = normalize(z_curated_tensor/255. , mean=torch.tensor(IMAGENET_DEFAULT_MEAN), std=torch.tensor(IMAGENET_DEFAULT_STD))
            template_curated_image_cache = z_curated.unsqueeze(0)
            template_image_mean = torch.mean(template_curated_image_cache,axis =2)
            template_image_mean = torch.mean(template_image_mean,axis =2)
            template_object_bbox =torch.tensor(box_corner)
            video_data_dic = {
                'z_curated' : z_curated,
                'z_bbox' : template_object_bbox,
                'z_image_mean' : template_image_mean,
                'x' : full_image,
                'frame_index' : 1,
                'z_feat' : None
                }
            video_data = RunTrackerData(video_data_dic)            
            break
        while True:
            with torch.no_grad():
                predicted_bounding_box = run_tracking(model,
                 video_data,
                 search_image_curation_parameter_provider,
                 search_curation_image_size,
                 bounding_box_post_processor,
                 post_processor,
                 interpolation_mode,
                 device)
                output_box = predicted_bounding_box[0].to(device = 'cpu').numpy()
                output_box_int = output_box.astype(int)
                state = [output_box_int[0],
                output_box_int[1],
                output_box_int[2]-output_box_int[0]+1,
                output_box_int[3]-output_box_int[1]+1]

                # show output
                cv.rectangle(frame, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                            (0, 255, 0), 5)
                font_color = (0, 0, 0)
                cv.putText(frame, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                        font_color, 1)
                cv.putText(frame, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                        font_color, 1)
                cv.putText(frame, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                        font_color, 1)

                # Display the resulting frame
                cv.imshow(display_name, frame)
                key = cv.waitKey(1)
                if key == ord('q'):
                    break
    
                #update image
                _, frame = cap.read()
                frame_RGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
                full_image = numpy_to_torch(frame_RGB).squeeze(0)
                full_image = full_image.to(device="cuda")
                video_data.full_image = full_image
                video_data.z_bbox = predicted_bounding_box.squeeze(0)
                video_data.frame_index += 1

def run_iteration(model, data_loader, runner, branch_name, event_dispatcher, logger, is_training, epoch):
    with enable_logger(logger), set_grad_enabled(is_training):
        runner.switch_branch(branch_name)
        runner.train(is_training)
        model.train(is_training)
        if not is_training:
            model = get_model(model)
        event_dispatcher.epoch_begin(epoch)
        for data in logger.loggers['local'].log_every(data_loader):
            seq_name = data[2]['seq_name'][0]  # prevent from collision
            event_dispatcher.iteration_begin(is_training)
            logger.set_step(runner.get_iteration_index())
            if seq_name == 'GOT-10k_Test_000152':
                z_curated = data[0]['z_curated'][0] # size[3,112,112] should be template
                x = data[0]['x'][0] #total image size[3,1080,1920]
                runner.run_iteration(model, data)
            event_dispatcher.iteration_end(is_training)
        event_dispatcher.epoch_end(epoch)
        gc.collect()
        epoch_status = event_dispatcher.collect_status()
        if epoch_status is not None and len(epoch_status) > 0:
            print(f'Epoch {epoch} branch {branch_name} statistics:')
            for status_name, status in epoch_status.items():
                print('----------------------------')
                print(f'{status_name}:')
                print(status)
                print('----------------------------')


class RunnerDriver:
    def __init__(self, name, n_epochs, model, runs, event_dispatcher, runtime_vars, efficiency_assessor, wandb_instance,
                 profiler, default_logger):
        self.name = name
        self.model = model
        self.event_dispatcher = event_dispatcher
        self.runtime_vars = runtime_vars
        self.n_epochs = n_epochs
        self.runs = runs
        self.wandb_instance = wandb_instance
        self.efficiency_assessor = efficiency_assessor
        self.epoch = 0
        if profiler is None:
            self.profiler = nullcontext()
        else:
            self.profiler = profiler
        self.dumping_interval = runtime_vars.checkpoint_interval
        self.default_logger = default_logger
        self.output_path = runtime_vars.output_dir
        self.resume_path = runtime_vars.resume
        self.device = torch.device(runtime_vars.device)

    def run(self):
        if self.resume_path is not None:
            model_state_dict, objects_state_dict = load_checkpoint(self.resume_path)
            assert model_state_dict['version'] == 2
            get_model(self.model).load_state_dict(model_state_dict['model'])
            self.epoch = objects_state_dict['epoch']
            self.event_dispatcher.dispatch_state_dict(objects_state_dict)

        self.event_dispatcher.device_changed(self.device)

        has_training_run = False
        for run in self.runs.values():
            is_training_run = run.is_training
            if is_training_run:
                has_training_run = True

        if has_training_run:
            print("Start training")
        else:
            print("Start evaluation")
        if self.wandb_instance is not None:
            _wandb_watch_model(get_model(self.model), self.wandb_instance, self.runtime_vars.watch_model_parameters,
                               self.runtime_vars.watch_model_gradients, self.runtime_vars.watch_model_freq)

        print_model_efficiency_assessment(self.efficiency_assessor, get_model(self.model), self.wandb_instance)

        start_time = time.perf_counter()

        if has_training_run:
            description = f'Train {self.name}'
        else:
            description = f'Evaluate {self.name}'
        with enable_logger(self.default_logger), self.event_dispatcher, self.profiler:
            for epoch in tqdm(range(self.epoch, self.n_epochs), desc=description, file=sys.stdout):
                print()
                self.epoch = epoch

                epoch_has_training_run = False
                for branch_name, (data_loader, runner, logger, is_training, epoch_interval, run_in_last_epoch, event_dispatcher) in self.runs.items():
                    assert epoch_interval >= 0
                    if is_training:
                        epoch_has_training_run = True
                    if (run_in_last_epoch and epoch + 1 == self.n_epochs) or (epoch_interval != 0 and epoch % epoch_interval == 0):
                        # run_iteration(self.model, data_loader, runner, branch_name, event_dispatcher, logger, is_training, epoch)
                        run_tracker(self.model,runner,branch_name)

                if epoch_has_training_run and self.output_path is not None and (epoch % self.dumping_interval == 0 or epoch + 1 == self.n_epochs):
                    model_state_dict = {'version': 2, 'model': get_model(self.model).state_dict()}
                    objects_state_dict = {'epoch': epoch}
                    additional_objects_state_dict = self.event_dispatcher.collect_state_dict()

                    if additional_objects_state_dict is not None:
                        objects_state_dict.update(additional_objects_state_dict)
                    dump_checkpoint(model_state_dict, objects_state_dict, epoch, self.output_path)

        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        if has_training_run:
            print(f'Training time {total_time_str}')
        else:
            print(f'Evaluating time {total_time_str}')
