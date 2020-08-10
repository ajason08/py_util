import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

# for reproducibility using seed_torch()
import os 
import random
#auxiliary packages
import time

def seed_torch(seed=1234, cudnn_benchmark=False):
    #This flag allows you to enable the inbuilt cudnn auto-tuner to find the 
    # best algorithm to use for your hardware
    # example for conv 
    # https://github.com/pytorch/pytorch/blob/1848cad10802db9fa0aa066d9de195958120d863/
    #aten/src/ATen/native/cudnn/Conv.cpp#L486-L494
    
    torch.backends.cudnn.benchmark = cudnn_benchmark
    
    torch.manual_seed(seed)  #defaul1
    torch.backends.cudnn.enabled = True
    
    ## other sources of randomness
    #torch.backends.cudnn.deterministic = True #defaul2        
    #torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed) #seed all gpus
    
    ## less probable sources of randomness 
    #np.random.seed(seed)
    #np.random.RandomState(seed)    
    #os.environ['PYTHONHASHSEED'] = str(seed)
    #random.seed(seed)  

from torchtext.data import Field, Dataset, Example
import pandas as pd

class DataFrameDataset(Dataset):
    """Class for using pandas DataFrames as a datasource"""
    def __init__(self, examples, fields, filter_pred=None):
        """
        Create a dataset from a pandas dataframe of examples and Fields
        Arguments:
            examples pd.DataFrame: DataFrame of examples
            fields {str: Field}: The Fields to use in this tuple. The
                string is a field name, and the Field is the associated field.
            filter_pred (callable or None): use only exanples for which
                filter_pred(example) is true, or use all examples if None.
                Default is None
        """
        self.examples = examples.apply(SeriesExample.fromSeries, args=(fields,), axis=1).tolist()
        if filter_pred is not None:
            self.examples = filter(filter_pred, self.examples)
        self.fields = dict(fields)
        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]

class SeriesExample(Example):
  """Class to convert a pandas Series to an Example"""

  @classmethod
  def fromSeries(cls, data, fields):
      return cls.fromdict(data.to_dict(), fields)

  @classmethod
  def fromdict(cls, data, fields):
    ex = cls()

    for key, field in fields.items():
        if key not in data:
            raise ValueError("Specified key {} was not found in "
            "the input data".format(key))
        if field is not None:
            setattr(ex, key, field.preprocess(data[key]))
        else:
            setattr(ex, key, data[key])
    return ex

def dataset_splitting(dfinput, text_col, label_col, path_input=True, splitting_col="splitting", sep="\t"):
    df = pd.read_csv(dfinput, sep) if path_input else dfinput
    rename_dict = {text_col:'text_field',
                  label_col:'label_field'}
    df.rename(columns=rename_dict, inplace=True)  
    #split datasets
    train_df = df[df[splitting_col] != "dev"]
    dev_df = df[df[splitting_col] == "dev"]
    #column filter
    field_names = list(rename_dict.values())
    train_df = train_df.filter(field_names)
    dev_df = dev_df.filter(field_names)
    return train_df, dev_df, field_names


def train_model(model, text_field, label_field, train_iterator, valid_iterator, optimizer, criterion,
                 saved_model_path, metric, avg, epochs, verbosity = False):  
  def epoch_time(elapsed_time):    
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

  metrics_report = ['loss', 'accuracy','precision', 'recall', 'fscore']   
  metric_types = ['loss', 'accuracy', 'fscore']  #'precision', 'recall', 'fscore']  
  if metric not in metric_types:
    raise ValueError(f'Invalid metric type. Expected one of {metric_types}')

  #freeze embeddings
  #model.embedding.weight.requires_grad = unfrozen = False  
  best_model_out = float('inf') if metric == "loss" else -1
  best_epoch = -1
  for epoch in range(epochs):
    start_time = time.time()
    #train_loss, train_acc = train(model, train_iterator, optimizer, criterion, avg)    
    #train_loss, train_acc, train_prec, train_rec, train_f1 = train(model, train_iterator, optimizer, criterion, avg)        
    #valid_loss, valid_acc, valid_prec, valid_rec, valid_f1 = evaluate(model, valid_iterator, criterion, avg)        
    train_scores = train(model, train_iterator, optimizer, criterion, avg)        
    valid_scores = evaluate(model, valid_iterator, criterion, avg)            
    epoch_mins, epoch_secs = epoch_time(time.time() - start_time)
    
    # valid_scores_switcher = {
    #   "loss": valid_loss,
    #   "accuracy": valid_acc,
    #   #"precision": valid_prec,
    #   #"recall": valid_rec,
    #   "fscore": valid_f1,
    # }

    train_results = dict(zip(metrics_report,train_scores))
    valid_results = dict(zip(metrics_report,valid_scores))

    #last_metric_out = valid_scores_switcher.get(metric)
    last_metric_out = valid_results.get(metric)
    better_loss = (metric == "loss" and last_metric_out < best_model_out)
    better_other_metric = (metric != "loss" and last_metric_out > best_model_out)
    if better_loss or better_other_metric:
      best_model_out = last_metric_out
      torch.save(model.state_dict(), saved_model_path)      
      best_epoch = epoch

    if verbosity:
      trainsr = pd.Series(train_results, name="train") 
      validsr = pd.Series(valid_results, name="valid") 
      epoch_report_df = pd.DataFrame(columns=metrics_report)
      epoch_report_df = epoch_report_df.append(trainsr).append(validsr)
      print(f'Epoch {epoch:02}; time: {epoch_mins}m {epoch_secs}s')
      display(epoch_report_df.round(decimals=6))
      # print(f'Epoch: {epoch+1:02}, Time: {epoch_mins}m {epoch_secs}s')    
      # print(
      #   f'\tTrain| Loss: {train_loss:.3f} | Acc: {train_acc*100:.2f}%'
      #   f'| prec: {train_prec:.3f} | rec: {train_rec:.3f} | f1: {train_f1:.3f}'   
      # )
      # print(
      #   f'\tValid| Loss: {valid_loss:.3f} | Acc: {valid_acc*100:.2f}% '
      #   f'| prec: {valid_prec:.3f} | rec: {valid_rec:.3f} | f1: {valid_f1:.3f}'   
      # )

    #### unfrezing embeddings after certain epoch
    #if epoch > 8: # it starts freezing from 10 (epochs starts at 0)
        #unfreeze embeddings
     #   model.embedding.weight.requires_grad = unfrozen = True
        
    output_tr_result = pd.DataFrame({
      'metric_focus' : [metric+"_"+avg],
      'best_result_tr' : [best_model_out],
      'best_epoch' : [best_epoch+1]
    })
  return output_tr_result

def test_model(model, test_iterator, saved_model_path, label_vocab, metric, avg, operations,
                 gold=True, verbosity=False, tr_result=None):
  """Perform predictions given a model; return human_label_prediction and a report if possible.
  the parameter operations receives a set considering these 
  values = {"confusion","class_metrics", "average_schemes"}.
  """
  model.load_state_dict(torch.load(saved_model_path))
  predictions, answers = predict_testset(model,iterator=test_iterator) 
  
  #(the original string labels)
  human_pred = [label_vocab.itos[pred_class] for pred_class in predictions]    
  report = None
  
  if gold: # using gold label we can make a report
    confusion, class_metrics, average_schemes = calculate_scores(answers, predictions, label_vocab,
                        operations = operations, verbosity=verbosity,
                        metric_focus = metric, metric_average=avg)      
    report = average_schemes.loc["fscore"]
    # add training information if possible
    if tr_result is not None: report["best_epoch"] = tr_result.loc[0,"best_epoch"]      
  return human_pred, report

############################################################## new methods
############################################################## new methods
############################################################## new methods

from IPython.display import Markdown, display
def calculate_scores(answers, predictions, labels, operations, verbosity=True,
                     metric_focus="fscore", metric_average="macro"):
  # to-doc: operations = {"confusion","class_metrics", "average_schemes"}
  
  metrics = ("precision", "recall", "fscore", "support")
  results = [None]*3
  if verbosity: print(f"{'-'*35} Evaluation summary:")
  labels_human = labels.itos
  # confusion matrix
  if "confusion" in operations:
    labels_values= list(labels.stoi.values())    
    cmatrix = confusion_matrix(answers, predictions, labels_values)
    confusion_df = pd.DataFrame(cmatrix, index=labels_human, columns=labels_human)
    confusion_df.columns.name = 'Gold \ Pred' # consider use too: confusion.index.name = 'Gold' 
    results[0] = confusion_df
    if verbosity: display(Markdown("**>Confusion matrix**:"), confusion_df)

  # Metrics by class
  if "class_metrics" in operations:    
#     scores = precision_recall_fscore_support(answers, predictions, average=None)
#     scores_dict = dict(zip(metrics, scores))  
#     class_metrics_df = pd.DataFrame(scores_dict, index = labels_human)
#     class_metrics_df.columns.name = 'class \ metric'
    rep_dict = classification_report(answers, predictions, output_dict=True)
    class_metrics_df = pd.DataFrame(rep_dict).filter(map(str, labels_human)).round(7).astype(object)
    class_metrics_df.columns.name = 'metric \ class'
    results[1] = class_metrics_df
    if verbosity: display(Markdown("**>Metrics by class**:"), class_metrics_df)

  # Average schemes comparison
  if "average_schemes" in operations:
    avgschemes =  ["micro", "macro", "weighted"]
    scores_avg = [precision_recall_fscore_support(answers, predictions, average=x)
                    for x in avgschemes]              
    scores_avg_dict = dict(zip(avgschemes, scores_avg))  
    average_schemes_df = pd.DataFrame(scores_avg_dict, index = metrics)
    average_schemes_df.columns.name = 'metric \ scheme'
    average_schemes_df = average_schemes_df.drop('support')
    results[2] = average_schemes_df
    if verbosity: display(Markdown("**>Average schemes comparison**:"),average_schemes_df)
  return results

def mlscoring(truth, preds, verbosity=True, avgScheme=None):
    """Returns ML scores using selected average scheme per batch."""
    #label_diffs()
    
    # get accuracy, i.e. if you get 8/10 right, this returns 0.8, NOT 8    
    max_preds = preds.argmax(dim = 1) # get the index of the max probability
    correct = max_preds.eq(truth)
    acc = correct.sum() / torch.FloatTensor([truth.shape[0]])    
    
    # get precision, recall, and f1 using sklearn (the support metric is not needed)
    preds_list = max_preds.cpu().tolist()
    truth_list = truth.cpu().tolist()
    
    return acc, precision_recall_fscore_support(truth_list, preds_list,
                                            average=avgScheme, zero_division=0)

def categorical_accuracy(preds, truth):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(truth)
    return correct.sum() / torch.FloatTensor([truth.shape[0]])

def _label_diffs(truth,pred):  
  difft = set(pred) - set(truth)
  diffp = set(truth) - set(pred)
  if len(difft): print(f'unexpected labels predicted: {difft}')
  if len(diffp): print(f'not predicted labels: {diffp}')  

# def train(model, iterator, optimizer, criterion,avg):
#     epoch_loss = 0
#     epoch_acc = 0    
#     model.train()

#     #predict for updating weights    
#     for batch in iterator:        
#         optimizer.zero_grad()        
        
#         predictions = model(batch.text_field)        
#         loss = criterion(predictions, batch.label_field)        
#         #acc = categorical_accuracy(predictions, batch.label_field)
#         acc, (batch_prec, batch_rec, batch_f1, _) = mlscoring(batch.label_field, predictions, avgScheme=avg)
        
#         loss.backward()        
#         optimizer.step()
        
#         epoch_loss += loss.item()
#         epoch_acc += acc.item()        
#     return epoch_loss / len(iterator), epoch_acc / len(iterator)
def train(model, iterator, optimizer, criterion,avg):
    epoch_loss = epoch_acc = 0
    epoch_prec = epoch_rec = epoch_f1 =0
    
    model.train()
    iter_len = len(iterator)    
    for batch in iterator:        
        optimizer.zero_grad()        
        
        predictions = model(batch.text_field)        
        loss = criterion(predictions, batch.label_field)        
        #acc = categorical_accuracy(predictions, batch.label_field)
        acc, (batch_prec, batch_rec, batch_f1, _) = mlscoring(batch.label_field, predictions, avgScheme=avg)
        
        loss.backward()        
        optimizer.step() #here updating weights?

        epoch_loss += loss.item()
        epoch_acc += acc.item()     
        epoch_prec += batch_prec
        epoch_rec += batch_rec
        epoch_f1 += batch_f1           
    epoch_loss = epoch_loss / iter_len   
    epoch_acc = epoch_acc / iter_len 
    epoch_prec = epoch_prec / iter_len     
    epoch_rec = epoch_rec / iter_len    
    epoch_f1 = epoch_f1 / iter_len 
    return epoch_loss, epoch_acc, epoch_prec, epoch_rec, epoch_f1

def evaluate(model, iterator, criterion, avg):   
    epoch_loss = epoch_acc = 0
    epoch_prec = epoch_rec = epoch_f1 =0

    model.eval()    
    iter_len = len(iterator)
    with torch.no_grad(): # predict without update weights
        for batch in iterator:

            predictions = model(batch.text_field)
            loss = criterion(predictions, batch.label_field)                                    
            acc, (batch_prec, batch_rec, batch_f1, _) = mlscoring(batch.label_field, predictions, avgScheme=avg)
            epoch_loss += loss.item()
            epoch_acc += acc.item()     
            epoch_prec += batch_prec
            epoch_rec += batch_rec
            epoch_f1 += batch_f1           
        epoch_loss = epoch_loss / iter_len   
        epoch_acc = epoch_acc / iter_len 
        epoch_prec = epoch_prec / iter_len     
        epoch_rec = epoch_rec / iter_len    
        epoch_f1 = epoch_f1 / iter_len 
    return epoch_loss, epoch_acc, epoch_prec, epoch_rec, epoch_f1

def predict_testset(model,iterator):  
  model.eval()
  mypredictions = []
  correct_answers = []
  with torch.no_grad(): # predict without update weights
    for batch in iterator:
      predictions = model(batch.text_field)            
      max_preds = predictions.argmax(dim = 1)
      mypredictions.append(max_preds)
      correct_answers.append(batch.label_field)
  mypredictions = torch.cat(mypredictions).tolist()
  correct_answers = torch.cat(correct_answers).tolist() 
  return mypredictions, correct_answers


def bert_tokenize_and_cut(bert_tokenizer):  
  def innerfunction(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:tokenizer.max_len-2]
    return tokens

def count_parameters(model, verbose=True):
    parcount = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose: print(f'The model has {parcount:,} trainable parameters')
    return parcount

