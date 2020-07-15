try:
    import mindspore
except Exception as e:
    raise ImportError(f"Mindspore were not installed correctly: {e}\n "
                      f"please go to https://www.mindspore.cn/install for more information")
try:
    import mindinsight
except Exception as e:
    raise ImportError(f"Mindinsight were not installed correctly: {e}\n "
                      f"please go to https://www.mindspore.cn/install#%E5%AE%89%E8%A3%85mindinsight for more information")
from setuptools import setup, find_packages

setup(
    name='SubgraphDetection',
    version='0.0.4',
    description=(
        'Detect the subgraph in a mindspore computational graph'
    ),
    author='loopyme',
    author_email='peter@mail.loopy.tech',
    maintainer='loopyme',
    maintainer_email='peter@mail.loopy.tech',
    license='MIT License',
    packages=find_packages("src"),
    package_dir={"": "src"},
    platforms=["all"],
    python_requires=">=3.6",
    entry_points={
        'console_scripts': [
            'detect-subgraph=SubgraphDetection:__main__.detect_subgraph_in_console',
        ]
    }
)
