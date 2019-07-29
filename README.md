# nbsvm-sklearn

A Scikit-learn wrapper for the NBSVM algorithm. To read more about the same, go [here](https://www.aclweb.org/anthology/P12-2018)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install nbsvm-sklearn.

```bash
pip install nbsvm-sklearn
```
Code has been tested on python>=3.6

## Usage

```python
from nbsvm import NBSVMClassifier

clf = NBSVMClassifier() # initialize the model
clf.fit(X, y) # train the classifier; y{0,1}
clf.predict(X) # get binary predictions

```
Full Code documentation available [here](nbsvm/nbsvm.py)

## Updates

Version 0.0.5: Added Platt-scaling

## Future Work
* Add support for multi-class classification
* Handle sparse matrices as inputs
* Handle ```pd.Series``` input format for labels

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
