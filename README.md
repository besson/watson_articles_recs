# Watson articles recommendations

Generating different type of recommendations of IBM Watson articles. This project is part of a Nanodegree in Data Science at [Udacity](https://www.udacity.com/) and it is structured as follows:

* **reports/**: It contains the main notebook (main deliverable) in the Markdow file which is easier to read than *.ipynb* format
* **data/**: Sample data for generating recommemndations
* **Recommendations_with_IBM.ipynb**: Notebook to explore and generate the recommendations
* **models.py**: Object models used to represent recommendations
* **article2vec.py**: Doc2vec Gensin model to generate content-based article recommendations
* **article2v.model**:  Serialized version of article2vec model
* **requirements.txt**: Main dependencies requirements
* **project_tests.py**: project validation tests
* **top*.p**: Serialized recommendations for top n articles
* **user_item_matrix.p**: Serialized user item matrics used in the Collaborative filtering recommenders