### **Computer Vision API V2**

<br>

API served on port `4040`

<br>

**Endpoints**

<pre>
1. /classify - returns highest confidence prediction label
2. /detect   - returns highest confidence bounding box and associated label
3. /segment  - returns list of labels and base64 encoded image data
4. /remove   - returns base64 encoded image data without the background
5. /replace  - returns base64 encoded image data with the replaced background
6. /depth    - base64 encoded depth image data
7. /face     - returns detection bounding boxes
</pre>
