from setuptools import setup, find_packages

setup(
    name='csv_object_list_dataset_loader',
    version='1.1',
    description='Reads in .csv files in either TAF, \
        INTERACTION or inD format and creates an object.',
    author='Max Zipfl',
    author_email='zipfl@fzi.de',
    install_requires=['pandas', 'numpy'],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6",
    zip_safe=False
)
#
