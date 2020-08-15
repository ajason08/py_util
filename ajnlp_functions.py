import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

# for reproducibility using seed_torch()
import os 
import random
#auxiliary packages
import time

def seed_torch(seed=1234, cudnn=True, benchmark=False):
    # benchmark flag allows you to enable the inbuilt cudnn auto-tuner to find the 
    # best algorithm to use for your hardware
    # example for conv
    # https://github.com/pytorch/pytorch/blob/1848cad10802db9fa0aa066d9de195958120d863/
    #aten/src/ATen/native/cudnn/Conv.cpp#L486-L494
        
    torch.manual_seed(seed)  # defaul1
    torch.backends.cudnn.deterministic = True  # defaul2
    torch.backends.cudnn.enabled = cudnn
    torch.backends.cudnn.benchmark = benchmark
    #torch.cuda.manual_seed(seed) # not sure what this does
    
    ## less probable sources of randomness
    #np.random.seed(seed)
    #np.random.RandomState(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed) # commonly used for hash algorithms
    #random.seed(seed)  
    
    # seed all gpus # done by default from pytorch 0.3
    # https://discuss.pytorch.org/t/random-seed-initialization/7854/4
    #torch.cuda.manual_seed_all(seed) 

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


def train_model(model, train_iterator, valid_iterator, input_f, output_f, optimizer, criterion,
                 saved_model_path, selector, avg, epochs, verbosity = False):  
# def train_model(model, train_iterator, valid_iterator, optimizer, criterion,
#                  saved_model_path, selector, avg, epochs, verbosity = False):  
  def epoch_time(elapsed_time):    
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

  #metrics_report = ['loss', 'accuracy','precision', 'recall', 'fscore']   
  metrics_report = ['loss', 'accuracy','precision', 'recall', 'fscore']   
  selectors = ['loss', 'accuracy', 'fscore']  #'precision', 'recall', 'fscore']  
  if selector not in selectors:
    raise ValueError(f'Invalid metric type. Expected one of {selectors}')

  #freeze embeddings
  #model.embedding.weight.requires_grad = unfrozen = False  
  best_model_out = float('inf') if selector == "loss" else -1
  best_epoch = -1
  for epoch in range(epochs):
    start_time = time.time()
    
    train_scores = train(model, train_iterator, input_f, output_f, optimizer, criterion, avg)        
    valid_scores = evaluate(model, valid_iterator, input_f, output_f, criterion, avg)            

    # train_scores = train(model, train_iterator, optimizer, criterion, avg)        
    # valid_scores = evaluate(model, valid_iterator, criterion, avg)            
    epoch_mins, epoch_secs = epoch_time(time.time() - start_time)

    train_results = dict(zip(metrics_report,train_scores))
    valid_results = dict(zip(metrics_report,valid_scores))

    epoch_result = valid_results.get(selector)
    better_model_min = (selector == "loss" and epoch_result < best_model_out)
    better_model_max = (selector != "loss" and epoch_result > best_model_out)
    if better_model_min or better_model_max:
      best_model_out = epoch_result
      torch.save(model.state_dict(), saved_model_path)      
      best_epoch = epoch

    if verbosity:
      trainsr = pd.Series(train_results, name="train") 
      validsr = pd.Series(valid_results, name="valid") 
      epoch_report_df = pd.DataFrame(columns=metrics_report)
      epoch_report_df = epoch_report_df.append(trainsr).append(validsr)
      print(f'Epoch {epoch:02}; time: {epoch_mins}m {epoch_secs}s')
      display(epoch_report_df.round(decimals=6))

    #### unfrezing embeddings after certain epoch
    #if epoch > 8: # it starts freezing from 10 (epochs starts at 0)
        #unfreeze embeddings
     #   model.embedding.weight.requires_grad = unfrozen = True
        
    training_output = pd.Series({
      'epoch_result' : best_model_out,
      'best_epoch' : best_epoch+1
    })
  return training_output


def test_model(model, test_iterator, input_f, output_f, saved_model_path, label_vocab, temp_path,
              operations, gold=True, verbosity=False):
# def test_model(model, test_iterator, saved_model_path, label_vocab, temp_path,
#               operations, gold=True, verbosity=False):
  """Perform predictions given a model; return human_label_prediction and a report if possible.
  the parameter operations receives a set considering these 
  values = {"confusion","class_metrics", "average_schemes"}.
  """
  model.load_state_dict(torch.load(saved_model_path))
  predictions, answers = predict_testset(model, test_iterator, input_f, output_f) 
  #predictions, answers = predict_testset(model,iterator=test_iterator) 
  
  #(the original string labels)
  human_pred = [label_vocab.itos[pred_class] for pred_class in predictions]    
  report = None
  
  if gold: # using gold label we can make a report
    scores = calculate_scores(answers, predictions, label_vocab, temp_path,
                              operations = operations, verbosity=verbosity)      
    report = scores[2].loc["fscore"] # average_schemes
  return report, human_pred

############################################################## new methods
############################################################## new methods

from IPython.display import Markdown, HTML, display
def calculate_scores(answers, predictions, labels, temp_path, operations, verbosity=True):
  # to-doc: operations = {"confusion","class_metrics", "average_schemes"}
  def label_diffs(truth,pred):  
    difft = set(pred) - set(truth)
    diffp = set(truth) - set(pred)
    if len(difft): print(f'unexpected labels predicted: {difft}')
    if len(diffp): print(f'not predicted labels: {diffp}')  

  metrics = ("precision", "recall", "fscore", "support")
  results = [None]*3
  labels_human = labels.itos
  # confusion matrix
  if "confusion" in operations:
    labels_values= list(labels.stoi.values())    
    cmatrix = confusion_matrix(answers, predictions, labels=labels_values)
    confusion_df = pd.DataFrame(cmatrix, index=labels_human, columns=labels_human)
    confusion_df.columns.name = 'Gold \ Pred' # consider use too: confusion.index.name = 'Gold' 
    results[0] = confusion_df
    if verbosity: 
      try:
        assert len(labels_human)<20, "Too large confusion matrix, saved to file"
        display(Markdown("**> Confusion matrix**:"), HTML(confusion_df.to_html()))
      except:
        confusion_df.to_csv(temp_path+"confusion_matrix", sep="\t")
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
    if verbosity: 
      label_diffs(answers, predictions)
      display(Markdown("**> Metrics by class**:"), HTML(class_metrics_df.to_html()))
      #if len(labels_human)<20: display(Markdown("**>Metrics by class**:"), HTML(class_metrics_df.to_html()))
      #else: class_metrics_df.to_csv(temp_path+"class_metrics", sep="\t")      
      

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
    if verbosity: display(Markdown("**> Average schemes comparison**:"),average_schemes_df)
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

def _label_diffs(truth,pred):  
  difft = set(pred) - set(truth)
  diffp = set(truth) - set(pred)
  if len(difft): print(f'unexpected labels predicted: {difft}')
  if len(diffp): print(f'not predicted labels: {diffp}')  

def train(model, iterator, input_f:list, output_f:str, optimizer, criterion,avg):
#def train(model, iterator, optimizer, criterion,avg):
    epoch_loss = epoch_acc = 0
    epoch_prec = epoch_rec = epoch_f1 =0
    
    model.train()
    iter_len = len(iterator)
    for batch in iterator:        
        optimizer.zero_grad()
        # the att .text_field and .label_field came from Dataset(pytorch class)
        # (also see dataset_splitting function)        
        batch_inputs = [getattr(batch,in_f) for in_f in input_f]
        batch_labels = getattr(batch,output_f)
        predictions = model(*batch_inputs)        
        loss = criterion(predictions, batch_labels)
        acc, (batch_prec, batch_rec, batch_f1, _) = mlscoring(batch_labels, predictions, avgScheme=avg)
        #predictions = model(batch.text_field)
        #loss = criterion(predictions, batch.label_field)
        #acc, (batch_prec, batch_rec, batch_f1, _) = mlscoring(batch.label_field, predictions, avgScheme=avg)
        
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

def evaluate(model, iterator, input_f, output_f, criterion, avg):   
#def evaluate(model, iterator, criterion, avg):   
    epoch_loss = epoch_acc = 0
    epoch_prec = epoch_rec = epoch_f1 =0

    model.eval()    
    iter_len = len(iterator)
    with torch.no_grad(): # predict without update weights
        for batch in iterator:
            batch_inputs = [getattr(batch,in_f) for in_f in input_f]
            batch_labels = getattr(batch,output_f)
            predictions = model(*batch_inputs)        
            loss = criterion(predictions, batch_labels)
            acc, (batch_prec, batch_rec, batch_f1, _) = mlscoring(batch_labels, predictions, avgScheme=avg)
            # predictions = model(batch.text_field)
            # loss = criterion(predictions, batch.label_field)                                    
            # acc, (batch_prec, batch_rec, batch_f1, _) = mlscoring(batch.label_field, predictions, avgScheme=avg)
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

def predict_testset(model,iterator, input_f, output_f):
#def predict_testset(model,iterator):  
  model.eval()
  mypredictions = []
  correct_answers = []
  with torch.no_grad(): # predict without update weights
    for batch in iterator:
      batch_inputs = [getattr(batch,in_f) for in_f in input_f]
      batch_labels = getattr(batch,output_f)
      predictions = model(*batch_inputs)

      #predictions = model(batch.text_field)
      max_preds = predictions.argmax(dim = 1)
      mypredictions.append(max_preds)
      
      correct_answers.append(batch_labels)
      #correct_answers.append(batch.label_field)
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

