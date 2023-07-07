from setuptools import setup, find_packages

setup(
    name='quests',
    version='0.1.0',
    packages=find_packages(include=["quests"]),
    scripts=['scripts/emdxyz.py'],
)

