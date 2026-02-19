echo "Starting Setup"

echo "update apt"
apt update

echo "install unzip and vim"
apt install -y unzip
apt install -y vim

echo "installing pandas tqdm"
pip install pandas tqdm timm albumentations scikit-learn

mkdir ~/coner-vision/models

echo "install data files"
wget -P ~/coner-vision -O ~/coner-vision/data.zip "https://dhttps://drivendata-prod.s3.amazonaws.com/data/87/public/competition_VfIpjyh.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYSN7TAHVS%2F20260219%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260219T204258Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=c9cfd994863664baa0b97dff75011d9150e8edad55666ad6d5d954b1487b629drivenhttps://drivendata-prod.s3.amazonaws.com/data/87/public/competition_VfIpjyh.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYSN7TAHVS%2F20260219%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260219T163705Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=6bc62145353472e3b82bc577778d6842c754fe6aae6d571dc2176b9ed3f3316bdata-prod.s3.amazonaws.com/data/87/public/competition_VfIpjyh.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYSN7TAHVS%2F20260218%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260218T092425Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=e706f7852fbeb5c71ecf0c714855ebe39d50471a64460ab610d26c6aa68cc3af"
mkdir ~/coner-vision/data
unzip ~/coner-vision/data.zip -d ~/coner-vision/data
rm -rf ~/coner-vision/data.zip