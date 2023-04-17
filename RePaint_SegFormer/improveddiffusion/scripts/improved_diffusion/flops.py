from ptflops import get_model_complexity_info
from swinunet import SwinUNetModel

def prepare_input(resolution):
    x1 = torch.FloatTensor( 3, 256, 256)
    x2 = torch.FloatTensor(1)
    return dict(x = [x1, x2])

m = SwinUNetModel(in_channels=3,model_channels=128,out_channels=6)
flops, params = get_model_complexity_info(model, (3, 256, 256),input_constructor=prepare_input, as_strings=True,
                                          print_per_layer_stat=False)  # 不用写batch_size大小，默认batch_size=1
# print("Swin_unet")
print('Flops:  ' + flops)
print('Params: ' + params)
