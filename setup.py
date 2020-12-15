import setuptools
import pathlib

setuptools.setup(
    name="thuface",
    version=0.1,
    description="THU Face Expression Recognizer",
    author="Xia Zhiyi",
    author_email="1399250123@qq.com",
    url="https://github.com/xiazhiyi99/face-expression-recognization-deployment",
    packages=["thuface", "thuface.model"],
    include_package_data=True,
    #install_requires=["torch", "opencv-python", "numpy==1.16.4", "torchvision", "pillow"]
)