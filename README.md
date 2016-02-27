DetectText
==========

Detect text with stroke width transform.

## Dependencies
OpenCV 2.4+, boost.

## Compile

    g++ -o DetectText TextDetection.cpp FeaturesMain.cpp -lopencv_core -lopencv_highgui -lopencv_imgproc -I/path/to/current/directory

where /path/to/current/directory is replaced with the absolute path to the current directory.

### Using CMake

    mkdir build
    cd build
    cmake ..
    make

## To run
./TextDetection input_file output_file dark_on_light
where dark_on_light is 1 or 0, indicating whether the text is darker or lighter than the background.

## More 
Details on the algorithm can be found in:
http://www.cs.cornell.edu/courses/cs4670/2010fa/projects/final/results/group_of_arp86_sk2357/Writeup.pdf