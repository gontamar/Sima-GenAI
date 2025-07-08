# SiMa.ai GenAI demo package

This package is used to run the SiMa.Ai genai demo frontend. The package runs a webserver which 
interacts with the SiMa.ai modalix board to run the GenAI demo.

The package contains

* Python webserver code (app.py)
* Lots of testing code (under test/)
* Static templates (templates) for web rendering.
* Javascript files to record voice etc. 
* SSL certs used by the application. Media upload/download works with *https* only.

**PS: The demo works with Google Chrome browser only**

# To install the package.

Once the package is downloaded from artifactory

```
    $ tar xvf genai-demo.tar.gz
    $ ls genai-demo-app/
         + certs
         + assets
         + static
         + templates
         requirements.txt
         app.py
         + test
         README.md

    $ python3 -m venv demo
    
    $ source demo/bin/activate
    
    $ pip3 install -r requirements.txt
    
    $ python3 app.py --camidx 0 --ip 192.168.1.20 
    
    # The options camidx is the index of the camera enumerated by Linux, 
      and ip is the ipaddress of the mlsoc board

    # Check for server.log for any errors, or wait for the log
    
    # Browse the URL https://127.0.0.1:5000/ using GOOGLE CHROME ONLY
```

# For any queries, please write to 

Ashok Sudarsanam <ashok.sudarsanam@sima.ai>
Vimal Nakum      <vimal.nakum@sima.ai>
