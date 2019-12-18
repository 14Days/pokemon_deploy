import io
import pathlib
import torch
from PIL import Image
from app.model.model import VGG
from app.model.tag import color_tag, db_map
from app.model.transform import transforms


class Net:
    def __init__(self):
        file = pathlib.Path(__file__).parent.parent.parent
        file = pathlib.Path.joinpath(file, 'checkpoints', 'color.pkl')
        self.model = VGG('VGG16', len(color_tag.keys()))
        temp = torch.load(str(file), map_location=torch.device('cpu'))
        self.model.load_state_dict(temp['net'])
        self.model.eval()

    def prediction(self, image_bytes):
        temp = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_as_tensor = transforms(temp).unsqueeze(0)
        temp_label = self.model(image_as_tensor)
        _, y_hat = torch.max(temp_label.data, 1)

        # for item in color_tag.keys():
        #     if color_tag[item] == y_hat:
        #         return item

        tag_id = db_map.get(y_hat.numpy()[0])

        if tag_id is None:
            raise RuntimeError('没有匹配')

        return tag_id


signal_net = Net()
