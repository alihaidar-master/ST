from mimetypes import init
from data.tracking.post_processor.response_map import ResponseMapTrackingPostProcessing
from runners.interface import BaseRunner
from data.tracking.methods.sequential.curation_parameter_provider import SiamFCCurationParameterSimpleProvider
from data.tracking.methods.SiamFC.common.siamfc_curation import do_SiamFC_curation
import torch
from data.tracking.post_processor.bounding_box.default import DefaultBoundingBoxPostProcessor 
from data.operator.bbox.spatial.vectorized.torch.utility.normalize import BoundingBoxNormalizationHelper
from data.types.bounding_box_format import BoundingBoxFormat

def _run_fn(fn, args):
    if isinstance(args, (list, tuple)):
        return fn(*args)
    elif isinstance(args, dict):
        return fn(**args)
    else:
        return fn(args)



        
def run_tracking( model, TrackerData,search_image_curation_parameter_provider,search_curation_image_size,bounding_box_post_processor,post_processor,interpolation_mode,device ):
    template_image_mean = TrackerData.z_image_mean# data['z_image_mean']
    search_image = TrackerData.full_image# data['x']
    search_image = search_image.to(device=device)

    if TrackerData.frame_index == 1 :
        tracker_initialization_results = None
        template_object_bbox = TrackerData.z_bbox
        template_curated_image = TrackerData.z_curated
        template_curated_image = template_curated_image.to(device = device)
        search_image_curation_parameter_provider.initialize(template_object_bbox) # # update it with new position (output)
        template_curated_image_cache = template_curated_image.unsqueeze(0)
        initialization_samples = template_curated_image_cache
        tracker_initialization_results = _run_fn(model.initialize, initialization_samples)
        TrackerData.z_feat = tracker_initialization_results

    search_image_size = search_image.shape[1:]
    search_image_size = torch.tensor((search_image_size[1], search_image_size[0]),device=device)  # (W, H)
    curation_parameter = search_image_curation_parameter_provider.get(search_curation_image_size)
    search_curated_image_cache,_ = do_SiamFC_curation(search_image, search_curation_image_size, curation_parameter,
                            interpolation_mode, template_image_mean)
    outputs = None

    tracking_samples = {
        'z_feat' : TrackerData.z_feat,
        'x' : search_curated_image_cache.unsqueeze(0).to(device=device)
        }
        
    if tracking_samples is not None:
        outputs = _run_fn(model.track, tracking_samples)
        outputs = post_processor(outputs)
        predicted_iou, predicted_bounding_boxes = outputs['conf'], outputs['bbox']
        predicted_bounding_boxes = predicted_bounding_boxes.to(torch.float64)
        curation_parameter = curation_parameter.unsqueeze(0).to(device=device)
        predicted_bounding_boxes = bounding_box_post_processor(predicted_bounding_boxes, curation_parameter[:predicted_bounding_boxes.shape[0], ...])
    search_image_curation_parameter_provider.update(predicted_iou, predicted_bounding_boxes.squeeze(0), search_image_size)
    return predicted_bounding_boxes