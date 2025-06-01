import torch
from model.u2net import U2NET  # 또는 U2NETP
import os

def convert_pth_to_onnx(pth_path, onnx_path="u2net_pill_122000.onnx", input_size=(1, 3, 320, 320)):
    # 1. 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 모델 로드
    model = U2NET(3, 1)  # U2NETP(3,1)일 수도 있음
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. 더미 입력 텐서
    dummy_input = torch.randn(input_size).to(device)

    # 4. ONNX로 변환
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        do_constant_folding=True,
        verbose=True
    )

    print(f"✅ 변환 완료: {onnx_path}")

if __name__ == "__main__":
    convert_pth_to_onnx("C:\\Users\\HRILAB\\Desktop\\MEDISEG\\saved_models\\u2net\\u2net_bce_itr_122000_train_0.025590_tar_0.001964.pth")
