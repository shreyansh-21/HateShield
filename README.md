<h1>HateShield</h1>
<h2>Overview</h2>
<p><strong>HateShield</strong> is a deep learning-based system designed to detect hate speech in both text and images. It utilizes:</p>
<ul>
    <li><strong>LSTM (Long Short-Term Memory)</strong> for analyzing text-based hate speech.</li>
    <li><strong>ResNet (Residual Neural Networks)</strong> for detecting hate content in memes and images.</li>
</ul>
<p>This repository contains the implementation of both models, trained on multiple hate speech datasets.</p>

<hr>

<h2>Table of Contents</h2>
<ol>
    <li><a href="#datasets">Datasets</a></li>
    <li><a href="#lstm">LSTM for Text-Based Hate Speech Detection</a></li>
    <li><a href="#resnet">ResNet for Image-Based Hate Speech Detection</a></li>
    <li><a href="#run">How to Run</a></li>
    <li><a href="#results">Results</a></li>
</ol>

<hr>

<h2 id="datasets">Datasets</h2>
<h3>Text-Based Datasets</h3>
<ul>
    <li><strong>HateXplain</strong>: A dataset that provides explanations along with hate speech classification.</li>
    <li>Dataset Location: <a href="https://huggingface.co/datasets/HateXplain">Hugging Face</a></li>
    <li><strong>Twitter & YouTube Hate Comments Dataset</strong>: Contains hate speech from social media platforms.</li>
    <li>Dataset Location: <a href="https://www.kaggle.com/datasets">Kaggle</a></li>
</ul>

<h3>Image-Based Datasets</h3>
<ul>
    <li><strong>Hateful Memes Dataset</strong>: A multimodal dataset for hate speech detection in memes.</li>
    <li>Dataset Location: <a href="https://ai.facebook.com/datasets/hateful-memes">Facebook AI</a></li>
</ul>

<hr>

<h2 id="lstm">LSTM for Text-Based Hate Speech Detection</h2>
<h3>What is LSTM?</h3>
<p><strong>LSTM (Long Short-Term Memory)</strong> is a type of recurrent neural network (RNN) that is particularly useful for processing sequential data, such as text. Unlike traditional RNNs, LSTMs can handle long-range dependencies, making them effective in understanding the context of words in sentences.</p>

<h3>How LSTM Helps in Text Analysis</h3>
<ul>
    <li><strong>Captures Context</strong>: LSTMs remember words from earlier in a sentence, making them ideal for detecting implicit hate speech.</li>
    <li><strong>Handles Long Sequences</strong>: Works well with long tweets, comments, and posts where context is crucial.</li>
    <li><strong>Mitigates Vanishing Gradient Problem</strong>: Unlike simple RNNs, LSTMs use gates to selectively store and forget information.</li>
</ul>

<h3>Implementation Details</h3>
<ul>
    <li><strong>Preprocessing</strong>:
        <ul>
            <li>Tokenization of text.</li>
            <li>Removal of stop words and special characters.</li>
            <li>Padding sequences for uniform input size.</li>
        </ul>
    </li>
    <li><strong>Model Architecture</strong>:
        <ul>
            <li>Embedding Layer: Converts words into dense vectors.</li>
            <li>LSTM Layer: Captures sequential dependencies.</li>
            <li>Dense Layer: Classifies text as hateful or non-hateful.</li>
        </ul>
    </li>
</ul>

<p><strong>Code Reference:</strong> Check the <code>Text_LSTM.ipynb</code> file for full implementation.</p>

<hr>

<h2 id="resnet">ResNet for Image-Based Hate Speech Detection</h2>
<h3>What is ResNet?</h3>
<p><strong>ResNet (Residual Networks)</strong> is a deep convolutional neural network (CNN) architecture that introduces residual learning to solve the problem of vanishing gradients in deep networks. It allows for efficient training of very deep models.</p>

<h3>How ResNet Helps in Image Analysis</h3>
<ul>
    <li><strong>Feature Extraction</strong>: Detects text, symbols, and offensive imagery in memes.</li>
    <li><strong>Deep Learning Performance</strong>: Prevents degradation in accuracy as networks get deeper.</li>
    <li><strong>Residual Connections</strong>: Helps in learning more complex patterns compared to traditional CNNs.</li>
</ul>

<h3>Implementation Details</h3>
<ul>
    <li><strong>Preprocessing</strong>:
        <ul>
            <li>Image resizing and normalization.</li>
            <li>Data augmentation to improve model generalization.</li>
        </ul>
    </li>
    <li><strong>Model Architecture</strong>:
        <ul>
            <li>Convolutional Layers: Extracts features from images.</li>
            <li>Residual Blocks: Helps in deeper learning without vanishing gradients.</li>
            <li>Fully Connected Layers: Classifies images as hateful or non-hateful.</li>
        </ul>
    </li>
</ul>

<p><strong>Code Reference:</strong> Check the <code>ResNet_final.ipynb</code> file for full implementation.</p>

<hr>

<h2 id="run">How to Run</h2>
<h3>Prerequisites</h3>
<ul>
    <li>Python 3.8+</li>
    <li>TensorFlow</li>
    <li>Keras</li>
    <li>OpenCV</li>
    <li>Pandas</li>
    <li>scikit-learn</li>
    <li>Matplotlib</li>
</ul>

<h3>Steps to Run</h3>
<ol>
    <li>Clone the repository:
        <pre><code>git clone https://github.com/shreyansh-21/HateShield.git
cd HateShield</code></pre>
    </li>
    <li>Install dependencies:
        <pre><code>pip install -r requirements.txt</code></pre>
    </li>
    <li>Run the LSTM model:
        <pre><code>jupyter notebook
# Open and run Text_LSTM.ipynb</code></pre>
    </li>
    <li>Run the ResNet model:
        <pre><code>jupyter notebook
# Open and run ResNet_final.ipynb</code></pre>
    </li>
</ol>

<hr>

<h2 id="results">Results</h2>
<p>The models achieve the following performance metrics:</p>
<ul>
    <li><strong>LSTM Model</strong>:
        <ul>
            <li>Precision: <strong>96%</strong></li>
            <li>Recall: <strong>99%</strong></li>
            <li>F1-score: <strong>96%</strong></li>
        </ul>
    </li>
    <li><strong>ResNet Model</strong>:
        <ul>
            <li>Accuracy: <strong>97%</strong></li>
        </ul>
    </li>
</ul>
<hr>
