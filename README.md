# Tutorial to get you started with QWEN2.5VL models with flash-attn support on windows.

## Installation on windows

### Steps

hf page : https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
pip install git+https://github.com/huggingface/transformers accelerate

pip install qwen-vl-utils[decord]==0.0.8


pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124


https://github.com/kingbri1/flash-attention/releases
"C:\Users\flash_attn-2.7.4.post1+cu124torch2.6.0cxx11abiFALSE-cp311-cp311-win_amd64.whl"


https://github.com/woct0rdho/triton-windows/releases
"C:\Users\triton-3.2.0-cp311-cp311-win_amd64.whl"

bits and bytes
python -m pip install bitsandbytes --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui

huggingface-cli login
<token>
