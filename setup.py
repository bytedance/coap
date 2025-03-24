from setuptools import setup


with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name="coap-torch",
    version="1.0.0",
    description="COAP: Memory-Efficient Training with Correlation-Aware Gradient Projection",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Jinqi Xiao, Shen Sang, Tiancheng Zhi, Jing Liu, Qing Yan, Yuqian Zhang, Linjie Luo, Bo Yuan",
    author_email="jinqi.xiao@rutgers.edu",
    url="https://github.com/bytedance/coap",
    packages=["coap_torch"],
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)