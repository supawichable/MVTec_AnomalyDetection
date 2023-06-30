import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__)))
from fastai.vision import *
from utils import *
from constants import THRESHOLDS

class AnomalyDetector:
    def __init__(self, model_path = os.path.abspath('ml/models/'), model_filename = 'export_centerloss.pkl',
     train_emb_path=os.path.abspath('ml/models/train_embs_centerloss.pt'), data_path = os.path.abspath('../uploads'), 
     threshold=THRESHOLDS['centerloss']):
        self.img_size = 224
        self.learn = load_learner(model_path, model_filename)
        self.train_embs = torch.load(train_emb_path)
        self.threshold = threshold
        self.data_path = data_path

    def embed(self, dataloader):
        embedding_model = body_feature_model(self.learn.model)
        return get_embeddings(embedding_model, dataloader, return_y=False)

    def get_prediction(self, img_name_lst):
        dataloader = self.get_dataloader(img_name_lst)
        embeded = self.embed(dataloader)
        distances = n_by_m_distances(embeded, self.train_embs, how='cosine')
        preds = np.min(distances, axis=1)
        return np.where(preds > self.threshold, 1, 0)

    def get_dataloader(self, img_name_lst):
        labels = [0] * len(img_name_lst)
        tmp_data = ImageDataBunch.from_lists(self.data_path, img_name_lst, labels, valid_pct=0, ds_tfms=None, size=224)
        dl = torch.utils.data.DataLoader(tmp_data.train_ds, batch_size=tmp_data.batch_size, shuffle=False)
        dl = DeviceDataLoader(dl, tmp_data.device)
        return dl

