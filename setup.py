#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=6.0', 'scikit-learn', 'keras', 'numpy', 'pandas']

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="BlackPoint-CX",
    author_email='msg.receiver.bp@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Project for Kaggle Competition (Quora Insincere Questions Classification) ",
    entry_points={
        'console_scripts': [
            'kg_quora_insincere_questions_classification=kg_quora_insincere_questions_classification.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='kg_quora_insincere_questions_classification',
    name='kg_quora_insincere_questions_classification',
    packages=find_packages(include=['kg_quora_insincere_questions_classification']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/BlackPoint-CX/kg_quora_insincere_questions_classification',
    version='0.1.0',
    zip_safe=False,
)
