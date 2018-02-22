import torch
import pickle 
from torch.autograd import Variable 
from torchvision import transforms 
from model import ResidualBlock, AttnEncoder, AttnDecoderRnn
from PIL import Image



class ImageCaptioningSample():
	def __init__(self):
		super(ImageCaptioningSample, self).__init__()
		self.embed_size = 128 
		self.feature_size = 128 
		self.hidden_size = 256 
		self.num_layers = 1 

		self.encoder_path = 'data/caption_data/encoder-100-600.pkl'
		self.decoder_path = 'data/caption_data/decoder-100-600.pkl'
		self.vocab_path = 'data/caption_data/piechart.pkl'

	def to_var(self,x, volatile=False):
	    if torch.cuda.is_available():
	        x = x.cuda()
	    return Variable(x, volatile=volatile)

	def load_image(self, image_path, transform):
		image = Image.open(image_path).convert('RGB')
		image = image.resize([64, 64], Image.LANCZOS)

		if transform is not None:
			image = transform(image).unsqueeze(0)

		return image

	def sample(self, img_dir):
		# Image preprocessing
	    transform = transforms.Compose([ 
	        transforms.ToTensor(), 
	        transforms.Normalize((0.033, 0.032, 0.033), 
	                             (0.027, 0.027, 0.027))])
	    # Load vocabulary wrapper
	    with open(self.vocab_path, 'rb') as f:
	    	vocab = pickle.load(f)	

	    # Build Models
	    encoder = AttnEncoder(ResidualBlock, [3, 3, 3])
	    encoder.eval()  # evaluation mode (BN uses moving mean/variance)
	    decoder = AttnDecoderRnn(self.feature_size, self.hidden_size, 
	                        len(vocab), self.num_layers)

	    # Load the trained model parameters
	    encoder.load_state_dict(torch.load(self.encoder_path))
	    decoder.load_state_dict(torch.load(self.decoder_path))

	    image = self.load_image(img_dir, transform)
	    image_tensor = self.to_var(image, volatile=True)

	    # If use gpu
	    if torch.cuda.is_available():
	        encoder.cuda()
	        decoder.cuda()

        # Generate caption from image
	    feature = encoder(image_tensor)
	    sampled_ids = decoder.sample(feature)
	    ids_arr = []
	    for element in sampled_ids: 
	        temp = element.cpu().data.numpy()
	        ids_arr.append(int(temp))

	    # Decode word_ids to words
	    sampled_caption = []
	    for word_id in ids_arr:
	        word = vocab.idx2word[word_id]
	        sampled_caption.append(word)
	        if word == '<end>':
	            break
	    sentence = ' '.join(sampled_caption)

	    return sentence
	    

# pie_caption = ImageCaptioningSample()
# pie_caption.sample('data')