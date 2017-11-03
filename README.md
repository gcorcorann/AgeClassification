# AgeClassification
Age Classification algorithm based on "Classification of Age Groups Based on
Facial Features Wen - Bing Horng, Cheng - Ping Lee and Chun - Wen Chen"

An age group classification system for gray-scale facial images. The
classifications are as follows: babies, young adults, middle-aged adults, and
old adults.
The process is divided into three phases: location, feature extraction, and age
classification. Based on the symmetry of human faces and the variation of gray
levels, the positions of eyes, noses, and mouth can be located by applying the
Sobel operator and region labeling. Two geometric features and three winkly
features from a facial image are then obtain. Finally, a classification method
is used for classification.
