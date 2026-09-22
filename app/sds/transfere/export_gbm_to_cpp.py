import joblib
import numpy as np

MODEL_PATH = "./models/bpp/uav_bandpass_model.joblib"
OUT_CPP = "generated_gbm.cpp"

clf = joblib.load(MODEL_PATH)

n_trees = len(clf.estimators_)
n_features = clf.n_features_in_

def export_tree(tree, tree_id):
    """
    Exportiert einen einzelnen DecisionTreeRegressor als C++-If-Blöcke.
    """
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    value = tree.tree_.value

    def recurse(node):
        if children_left[node] == -1:  # Leaf
            leaf_val = float(value[node][0][0])
            return f"return {leaf_val:.8f}f;"

        feat = feature[node]
        thr = threshold[node]

        return (
            f"if (x[{feat}] <= {thr:.8f}f) {{\n"
            f"    {recurse(children_left[node])}\n"
            f"}} else {{\n"
            f"    {recurse(children_right[node])}\n"
            f"}}"
        )

    code = f"float tree_{tree_id}(const float x[10]) {{\n"
    code += "    " + recurse(0).replace("\n", "\n    ")
    code += "\n}\n\n"
    return code


# ---------------------------------------------------------
# Export aller Trees
# ---------------------------------------------------------
cpp_code = "// AUTO-GENERATED GBM MODEL\n\n"
cpp_code += "#include <cmath>\n\n"

for i in range(n_trees):
    tree = clf.estimators_[i, 0]
    cpp_code += export_tree(tree, i)

# ---------------------------------------------------------
# Ensemble-Funktion
# ---------------------------------------------------------
cpp_code += "float evalTrees(const float x[10]) {\n"
cpp_code += "    float score = 0.0f;\n"

for i in range(n_trees):
    cpp_code += f"    score += tree_{i}(x);\n"

cpp_code += "    return score;\n"
cpp_code += "}\n"

# ---------------------------------------------------------
# Datei schreiben
# ---------------------------------------------------------
with open(OUT_CPP, "w") as f:
    f.write(cpp_code)

print(f"[INFO] GBM exportiert nach: {OUT_CPP}")
