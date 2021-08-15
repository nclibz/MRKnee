# %%
from submit_utils import gather_preds
import pandas as pd
from joblib import load
import sys

# output 3 col csv no header: abnormality, ACL , meniscal tear


# load paths
def submit(input_path, output_path):
    df = pd.read_csv(input_path, header=None)

    all_preds = [gather_preds(df, plane) for plane in ['axial', 'sagittal', 'coronal']]

    probas = []
    for i, clf in enumerate(['abn_clf', 'acl_clf', 'men_clf']):
        preds = pd.DataFrame([preds[i] for preds in all_preds]).T
        clf = load(f'src/models/{clf}.joblib')
        probas.append(clf.predict_proba(preds)[:, 1])

    proba_out = pd.DataFrame(probas).T

    proba_out.to_csv(output_path, header=None, index=False)


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    submit(input_path, output_path)


# 'submit_test/valid-paths.csv'
# 'submit_test/output.csv'
