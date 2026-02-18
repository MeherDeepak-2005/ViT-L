echo "Starting Setup"

echo "update apt"
apt update

echo "install unzip and vim"
apt install -y unzip
apt install -y vim

echo "installing pandas tqdm"
pip install pandas tqdm

mkdir ~/coner-vision/models

echo "install data files"
wget -P ~/coner-vision -O ~/coner-vision/data.zip "https://drivendata-prod.s3.amazonaws.com/data/87/public/competition_VfIpjyh.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYSN7TAHVS%2F20260218%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260218T092425Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=e706f7852fbeb5c71ecf0c714855ebe39d50471a64460ab610d26c6aa68cc3af"
mkdir ~/coner-vision/data
unzip ~/coner-vision/data.zip -d ~/coner-vision/data
rm -rf ~/coner-vision/data.zip

echo "======== Executing ViT Model ==========="
cd ~/coner-vision/
python model.py --data_dir=./data --batch_size=128 --img_size=224