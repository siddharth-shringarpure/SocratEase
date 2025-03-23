# SocratEase
## SocratEase: Speak smarter with AI-powered feedback.

SocratEase is a magical platform designed to turn your speech videos into insightful feedback through advanced AI models. Whether you're preparing for a job interview, perfecting your public speaking skills, or mastering your first date conversations, SocratEase analyses your speech, facial expressions, and engagement levels to offer valuable, personalised feedback.

Using a combination of digital image processing, natural language processing (NLP), and audio signal processing, SocratEase evaluates your communication on three key dimensions: visual cues (like facial expressions and eye contact), auditory features (such as speech fluency), and textual analysis (for logical coherence and engagement). By combining these analyses through a late-fusion multimodal approach, SocratEase provides an integrated understanding of your speaking style and areas for improvement.

This approach, detailed in [D'Mello 2015](https://dl.acm.org/doi/pdf/10.1145/2682899), allows us to process each modality independently before combining them to generate actionable feedback; the final result is a comprehensive evaluation of your communication skills, helping you feel more confident and prepared for any speaking situation.

 Features

🎥 **Video Feed Analysis** – Evaluate user engagement through eye contact and facial expressions
   - 👀 **Eye Contact Detection** – Improve users' engagement in conversations and speeches using [GazeTracking](https://github.com/antoinelame/GazeTracking)'s method
   - 😀 **Facial Expression Analysis** – Identifies emotions and microexpressions by using techniques from [Edlitera](https://www.edlitera.com/blog/posts/emotion-detection-in-video)

🎙️ **Audio Feed Analysis** – Focuses on fluency features in speech.  
   - 🗣 **Fluency Metrics** – Implements techniques from [Eusipco 2023](https://eurasip.org/Proceedings/Eusipco/Eusipco2023/pdfs/0000231.pdf).  
- 📊 **Dataset Utilisation** – Uses the [Avalinguo-Audio-Set](https://github.com/agrija9/Avalinguo-Audio-Set).  
- 🔎 **Speech Features Extracted**:
    - ⏱ **Words per Minute**  
    - 📖 **Lexical Density** (Token Type Ratio)  
    - 🔕 **Zero Crossing Rate** (silent pauses)  
    - 🎵 **MFCC** (Mel-frequency cepstral coefficients)
    - The features are extracted, **trained on an extreme gradient boosting (XGB) model** [Chen 2016](https://dl.acm.org/doi/10.1145/2939672.2939785) to predict the **fluency**
        - The XGB model is trained using Randomised CV model selection, achieving 93% overall F1-Score
        - Alongside other features, fluency is also used as a feedback to the user

📝 **Communication Transcript Analysis** – Examines speech patterns, coherence, and engagement.  
   - ✍️ **Tonality Analysis** – Utilises [tone analysis dataset](https://www.kaggle.com/datasets/sameedatif/tone-analysis) + [
3gpp-embedding-model-v0](https://huggingface.co/iris49/3gpp-embedding-model-v0) + MLP
   - 🚫 **Filler Word Frequency** – Computes corpus occurrence in the speech's transcript.
   - 📚 **Vocabulary Sophistication** – Assesses Type Token Ratio (TTR) to see how much the words are repeated
   - 🔄 **Logical Flow Detection** – Leverages [roberta-large_overall-coherence](https://huggingface.co/SushantGautam/roberta-large_overall-coherence) to use logistic regression in finding the coherence of the speech
   - 🎭 **Engagement Prediction** – Uses **Flesch-Kincaid Readability** to estimate listener interests  
## Implementation
## Tech Stack
- Frontend: React, Next.js
- Backend: Flask, Python
- AI & CV & NLP & Signal: SVM, NLTK, re, Librosa, Torch, Neuphonic
- Integrations: NPM

## What’s Next for SocratEase?  
As we continue enhancing our analysis system, we plan to introduce new intelligent features and improvements, including:  

🌍 **Multilingual Speech & Text Support** – Expanding accessibility for diverse linguistic backgrounds.  

🎯 **Enhanced Emotion & Engagement Detection** – Refining sentiment analysis and listener interest prediction for more accurate insights.  

🎙️ **Real-Time Fluency Feedback** – Providing instant analysis of speech fluency with actionable recommendations.  

🕵️ **Context-Aware Coherence Evaluation** – Improving logical flow detection with more robust reasoning models.  

📊 **Comprehensive Communication Analytics** – Introducing detailed performance tracking and insights for continuous improvement.  
