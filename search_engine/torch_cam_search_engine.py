import torch
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms.functional import normalize, resize

from .opencv_back_sub_seach_engine import SearchEngineOpenCVBackgroundSubtractor
from image_with_name_dc import ImageWithName
from app_logger import get_logger

logger = get_logger(__name__)


class SearchEngineTorchCam(SearchEngineOpenCVBackgroundSubtractor):
    def __init__(self, threshold, **kwargs) -> None:
        model_name = kwargs.pop('model_name')
        super().__init__(threshold, **kwargs)
        if model_name == 'resnet18':
            self._model = resnet18(pretrained=True).eval()
        else:
            raise ValueError('')

    def run(self, obj):
        img = obj.img
        with SmoothGradCAMpp(self._model) as cam_extractor:
            img = torch.Tensor(img)
            resize_img = resize(img, (224, 224))
            input_tensor = normalize(resize_img / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # Preprocess your data and feed it to the model
            out = self._model(input_tensor.unsqueeze(0))
            # Retrieve the CAM by passing the class index and the model output
            activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

            res = super().run(ImageWithName(None, activation_map))
            if res is not None:
                return obj
            return None

    def run_batch(self, images_with_name):
        unique_images_w_n = []
        with SmoothGradCAMpp(self._model) as cam_extractor:
            for img_w_n in images_with_name:
                img = img_w_n.img
                img = torch.Tensor(img)
                resize_img = resize(img, (224, 224))
                input_tensor = normalize(resize_img / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                # Preprocess your data and feed it to the model
                out = self._model(input_tensor.unsqueeze(0))
                # Retrieve the CAM by passing the class index and the model output
                activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
                res = super().run(ImageWithName(None, activation_map))
                if res:
                    unique_images_w_n.append(img_w_n)
        return unique_images_w_n