#!/bin/bash

# 1. Update and install system utilities
echo "Updating system and installing utilities..."
sudo apt-get update
sudo apt-get install -y htop dstat python3-pip

# 2. Update PATH in .bashrc for the future
# We check if it's already there so we don't append it multiple times
if ! grep -q ".local/bin" ~/.bashrc; then
    echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
    echo "Added ~/.local/bin to PATH in .bashrc"
fi

# Manually update the current script's PATH since 'source' 
# inside a script doesn't always affect the calling shell
export PATH=$HOME/.local/bin:$PATH

# 3. Install PyTorch (CPU version as requested)
echo "Installing PyTorch (CPU)..."
pip3 install --user torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu

# 4. Install ML and Data Science stack
echo "Installing dependencies..."
pip3 install --user numpy scipy scikit-learn tqdm pytorch_transformers apex
pip3 install --user pandas matplotlib

echo "----------------------------------------"
echo "Setup complete! Please run 'source ~/.bashrc' manually now."
echo "----------------------------------------"

#!/bin/bash

REPO_URL="git@github.com:ke3574-ai/cos568_pa2"

# This pulls the name of the repo out of the URL (e.g., 'my-repo')
REPO_NAME=$(basename "$REPO_URL" .git)

echo "Cloning $REPO_NAME..."
git clone "$REPO_URL"

# Now you can use that variable to enter the folder
cd "$REPO_NAME"
ls -l

mkdir glue_data
python3 download_glue_data.py --data_dir glue_data

echo "Downloaded Glue Data!"