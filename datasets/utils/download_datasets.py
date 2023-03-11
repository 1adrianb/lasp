import os
from subprocess import call
import gdown
import argparse

args = argparse.ArgumentParser()
args.add_argument('--root', required=True, help='path  to datasets')
args.add_argument('--dataset_name', default=None, help='name of dataset to download. If not specified all datasets will be downloaded')

args = args.parse_args()

def download_datasets(root, dataset_names=None):
    if dataset_names is None:
        dataset_names = ['imagenet', 'caltech101', 'oxford_pets', 'stanford_cars', 'flowers102', 'food101',
                        'fgvc_aircraft', 'sun397', 'dtd', 'eurosat', 'ucf101']
    for dataset_name in dataset_names:
        download_dataset(dataset_name, f"{root}/{dataset_name}")


def download_dataset(dataset_name, root="root"):
    if dataset_name == 'imagenet':
        base_root = root
        root = os.path.join(root, 'images')
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
        else:
            return
    else:
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
        else:
            return

    if dataset_name == 'imagenet':    
        call(f"wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --output-document={root}/ILSVRC2012_img_val.tar", shell=True)
        call(f"wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --output-document={root}/ILSVRC2012_img_train.tar", shell=True)

        call(f"untar -xf {root}/ILSVRC2012_img_val.tar", shell=True)
        call(f"untar -xf {root}/ILSVRC2012_img_train.tar", shell=True)

        call(f"rm {root}/ILSVRC2012_img_val.tar", shell=True)
        call(f"rm {root}/ILSVRC2012_img_train.tar", shell=True)

        url = 'https://drive.google.com/uc?id=1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF'
        gdown.download(url, f"{base_root}/classnames.txt", quiet=False)
    elif dataset_name == "caltech101":
        call(f"wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz --output-document={root}/101_ObjectCategories.tar.gz", shell=True)
        call(f"untar -xf {root}/101_ObjectCategories.tar.gz", shell=True)
        call(f"rm {root}/101_ObjectCategories.tar.gz", shell=True)

        url = 'https://drive.google.com/uc?id=1hyarUivQE36mY6jSomru6Fjd-JzwcCzN'
        gdown.download(url, f"{root}/split_zhou_Caltech101.json", quiet=False)
    elif dataset_name == "oxford_pets":  
        call(f"wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz --output-document={root}/images.tar.gz", shell=True)
        call(f"wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz --output-document={root}/annotations.tar.gz", shell=True)
        call(f"untar -xf {root}/images.tar.gz", shell=True)
        call(f"untar -xf {root}/annotations.tar.gz", shell=True)
        call(f"rm {root}/images.tar.gz", shell=True)
        call(f"rm {root}/annotations.tar.gz", shell=True)

        url = 'https://drive.google.com/uc?id=1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs'
        gdown.download(url, f"{root}/split_zhou_OxfordPets.json", quiet=False)
    elif dataset_name == "stanford_cars":
        call(f"wget http://ai.stanford.edu/~jkrause/car196/cars_train.tgz --output-document={root}/cars_train.tgz", shell=True)
        call(f"wget http://ai.stanford.edu/~jkrause/car196/cars_test.tgz --output-document={root}/cars_test.tgz", shell=True)
        call(f"wget http://ai.stanford.edu/~jkrause/car196/cars_devkit.tgz --output-document={root}/cars_devkit.tgz", shell=True)
        call(f"wget http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat --output-document={root}/cars_test_annos_withlabels.mat", shell=True)
        call(f"untar -xf {root}/cars_train.tgz", shell=True)
        call(f"untar -xf {root}/cars_test.tgz", shell=True)
        call(f"untar -xf {root}/cars_devkit.tgz", shell=True)

        url = 'https://drive.google.com/uc?id=1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT'
        gdown.download(url, f"{root}/split_zhou_StanfordCars.json", quiet=False)
    elif dataset_name == "flowers102":
        call(f"wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz --output-document={root}/102flowers.tgz", shell=True)
        call(f"wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat --output-document={root}/imagelabels.mat", shell=True)

        call(f"untar -xf {root}/102flowers.tgz", shell=True)
        call(f"rm {root}/102flowers.tgz", shell=True)

        url = 'https://drive.google.com/uc?id=1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0'
        gdown.download(url, f"{root}/cat_to_name.json", quiet=False)

        url = 'https://drive.google.com/uc?id=1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT'
        gdown.download(url, f"{root}/split_zhou_OxfordFlowers.json", quiet=False)
    elif dataset_name == "food101":
        call(f"wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz --output-document={root}/food-101.tar.gz", shell=True)
        call(f"untar -xf {root}/food-101.tar.gz", shell=True)
        call(f"rm {root}/food-101.tar.gz", shell=True)

        url = 'https://drive.google.com/uc?id=1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl'
        gdown.download(url, f"{root}/split_zhou_Food101.json", quiet=False)
    elif dataset_name == "fgvc_aircraft":
        call(f"wget http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz --output-document={root}/fgvc-aircraft-2013b.tar.gz", shell=True)
        call(f"untar -xf {root}/fgvc-aircraft-2013b.tar.gz", shell=True)
        call(f"rm {root}/fgvc-aircraft-2013b.tar.gz", shell=True)
    elif dataset_name == "sun397":
        call(f"wget http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz --output-document={root}/SUN397.tar.gz", shell=True)
        call(f"wget https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip --output-document={root}/Partitions.zip", shell=True)
        call(f"untar -xf {root}/SUN397.tar.gz", shell=True)
        call(f"untar -xf {root}/Partitions.zip", shell=True)
        call(f"rm {root}/SUN397.tar.gz", shell=True)
        call(f"rm {root}/Partitions.zip", shell=True)

        url = 'https://drive.google.com/uc?id=1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq'
        gdown.download(url, f"{root}/split_zhou_SUN397.json", quiet=False)
    elif dataset_name == "dtd":
        call(f"wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz --output-document={root}/dtd-r1.0.1.tar.gz", shell=True)
        call(f"untar -xf {root}/dtd-r1.0.1.tar.gz", shell=True)
        call(f"rm {root}/dtd-r1.0.1.tar.gz", shell=True)

        url = 'https://drive.google.com/uc?id=1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x'
        gdown.download(url, f"{root}/split_zhou_DTD.json", quiet=False)
    elif dataset_name == "eurosat":
        call(f"wget http://madm.dfki.de/files/sentinel/EuroSAT.zip --output-document={root}/EuroSAT.zip", shell=True)
        call(f"untar -xf {root}/EuroSAT.zip", shell=True)
        call(f"rm {root}/EuroSAT.zip", shell=True)

        url = 'https://drive.google.com/uc?id=1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o'
        gdown.download(url, f"{root}/split_zhou_EuroSAT.json", quiet=False)
    elif dataset_name == "ucf101":
        url = 'https://drive.google.com/uc?id=10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O'
        gdown.download(url, f"{root}/UCF-101-midframes.zip", quiet=False)
        call(f"unzip {root}/UCF-101-midframes.zip -d {root}", shell=True)
        call(f"rm -rf {root}/UCF-101-midframes.zip", shell=True)

        url = 'https://drive.google.com/uc?id=1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y'
        gdown.download(url, f"{root}/split_zhou_UCF101.json", quiet=False)
    else:
        raise Exception('Unknown dataset.')
    
download_datasets(args.root)



