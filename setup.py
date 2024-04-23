from setuptools import setup

setup(
    name='daac',
    version='0.1',
    packages=['daac'],
    entry_points={
        'console_scripts': [
            'daac = daac.programs:main'
        ]
    }
)
