# Does-GenAI-Help
<img width="1024" height="768" alt="graphical_abstract_git" src="https://github.com/user-attachments/assets/0f077217-aa27-459f-9743-108a402d5d2a" />
This repository is built for the paper Why "Does GenAI Help?" Is the Wrong Question: Effects Change Sign Across Student Subgroups
## Overview
In this work, we instantiate PLS-SEM as a target model class within the EMM framework to discover student subgroups whose structural relationships between GenAI use and academic performance differ from the overall population. To improve interpretability, we complement EMM with a contrastive analysis that constructs minimally different comparison groups, enabling practitioners to understand which conditions drive the observed deviations.

As the first wave of a longitudinal study, we collected data in a large introductory data science course enrolling more than 1,000 first-year bachelor students across multiple study programs. We combine in-situ survey data on GenAI attitudes, skills, and usage behavior with exam outcomes to study how associations between GenAI usage modes and academic performance vary across students.

Here, we provide the [full surveys](surveys/), the [construct specifications](constructs/), and relevant parts of our source code. To deploy EMM, we extend the [pysubgroup](https://github.com/flemmerich/pysubgroup) (Lemmerich & Becker, 2018) package with an instantiation of [PLS-SEM as a target model class](PLS-SEM-extension/). This extension can be used on any dataset and with any PLS-SEM structure, provided that its configuration is specified using the [plspm](https://pypi.org/project/plspm/) package.

## Prerequisites & Installation
Install [pysubgroup](https://github.com/flemmerich/pysubgroup) and [plspm](https://pypi.org/project/plspm/), using pip:
```
pip install pysubgroup
pip install plspm
``` 
Note: for some users the pip installation of [pysubgroup](https://github.com/flemmerich/pysubgroup) fails. If that is the case, solutions are provided in their repository. 

Once installed:
1. Download the files in the [PLS-SEM-extension](PLS-SEM-extension/) folder;
2. Locate the folder `pysubgroup` within your `site-packages` directory;
3. Delete the `__init__.py` file;
4. Move the downloaded files to the `pysubgroup` folder;
5. You can now import the extended package with `import pysubgroup`.
## How To Use
