python3.7 -m venv ~/pyenv/fusion_in_decoder
source ~/pyenv/fusion_in_decoder/bin/activate
pip install -r ./requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
cd ./transformers-3.0.2
pip install -e .
cd ..
