apt install python3.7 python3.7-pip python3.7-dev
python3.7 -m pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.7.0-beta/MindSpore/ascend/ubuntu_x86/mindspore_ascend-0.7.0-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
python3.7 -m pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.7.0-beta/MindInsight/ascend/ubuntu_x86/mindinsight-0.7.0-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple

python3.7 setup.py install
rm test_result.csv
python3.7 grid_test.py