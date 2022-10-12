# Data Science Repository Template
For new data science projects, this repository should be cloned. It comes with the following pre-configured:

1. A `requirements.txt` file with the basic data science packages
2. Pre-commit hooks (see `.pre-commit-config.yaml` in repository)
3. A docker imager that can be configured [TBD]
4. A `.gitignore` that removes `data/`, `logs/`, and secrets files



Please see the [DS - Ways of Working](https://docs.google.com/document/d/118Bw155iqn9x7PQ25TlkH8FqKH8aHFidGYp3d0nD7k4/edit) for more information.



## Starting a new project

1. Click on the green `Use this template` button

2. Fill in repository name and description

3. Clone repository to local machine

4. Create a new conda environment using

   `conda create --name="<proj_name>" python==3.7`

5. Install packages

   `pip install -r requirements.txt`

6. Update the `README.md` file

7. Run `make directories` in the project root to create the standard directories

8. <STEPS FOR DOCKER>







