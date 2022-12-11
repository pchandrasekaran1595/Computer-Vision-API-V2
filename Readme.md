### **Computer Vision API V2**

<br>

API served on port `4040`

<br>

**Endpoints**

<pre>
1. `/classify`       - returns the highest confidence prediction label
2. `/detect`         - returns the highest confidence bounding box and associated label
3. `/segment`        - returns a list of labels and base64 encoded image data
4. `/remove`         - returns a base64 encoded image data without the background
5. `/replace`        - returns a base64 encoded image data with the replaced background
6. `/depth`          - returns a base64 encoded depth image data
7. `/face-detect`    - returns the detection bounding boxes
8. `/face-recognize` - returns the cosine similarity between two face images
</pre>
