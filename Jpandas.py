import pandas as pd
def sjoiner(x, *colnames):
"""
USAGE:
df["col1_col2"] = df.apply(sjoiner, args= ("col1","col2"), axis=1)
"""  
  values = [x[v] for v in colnames]
  return " ".join(values)

def row_splitter(table, splitting_col, filters:list, path_input=True, sep="\t"):
  """
  USAGE:
  train_df, dev_df = Jpandas.row_splitter(train_dev_file,
                                                "dataset", ["train","dev"],
                                                path_input=True, sep="\t")
  """  
  df = pd.read_csv(table, sep) if path_input else table
  return [df[df[splitting_col] == f] for f in filters]

