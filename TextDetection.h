/*
    Copyright 2012 Andrew Perrault and Saurav Kumar.

    This file is part of DetectText.

    DetectText is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    DetectText is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with DetectText.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef TEXTDETECTION_H
#define TEXTDETECTION_H

#include <opencv2/core/core.hpp>

namespace DetectText {

struct SWTPoint2d {
    int x;
    int y;
    float SWT;
};

typedef std::pair<SWTPoint2d, SWTPoint2d> SWTPointPair2d;
typedef std::pair<cv::Point, cv::Point>   SWTPointPair2i;

struct Point2dFloat {
    float x;
    float y;
};

struct Ray {
        SWTPoint2d p;
        SWTPoint2d q;
        std::vector<SWTPoint2d> points;
};

struct Point3dFloat {
    float x;
    float y;
    float z;
};


struct Chain {
    int p;
    int q;
    float dist;
    bool merged;
    Point2dFloat direction;
    std::vector<int> components;
};

bool Point2dSort (SWTPoint2d const & lhs,
                  SWTPoint2d const & rhs);

cv::Mat textDetection (const cv::Mat& input, bool dark_on_light);

void strokeWidthTransform (const cv::Mat& edgeImage,
                           cv::Mat& gradientX,
                           cv::Mat& gradientY,
                           bool dark_on_light,
                           cv::Mat& SWTImage,
                           std::vector<Ray> & rays);

void SWTMedianFilter (cv::Mat& SWTImage, std::vector<Ray> & rays);

std::vector< std::vector<SWTPoint2d> > findLegallyConnectedComponents (cv::Mat& SWTImage, std::vector<Ray> & rays);

std::vector< std::vector<SWTPoint2d> >
findLegallyConnectedComponentsRAY (IplImage * SWTImage,
                                std::vector<Ray> & rays);

void componentStats(IplImage * SWTImage,
                                        const std::vector<SWTPoint2d> & component,
                                        float & mean, float & variance, float & median,
                                        int & minx, int & miny, int & maxx, int & maxy);

void filterComponents(cv::Mat& SWTImage,
                      std::vector<std::vector<SWTPoint2d> > & components,
                      std::vector<std::vector<SWTPoint2d> > & validComponents,
                      std::vector<Point2dFloat> & compCenters,
                      std::vector<float> & compMedians,
                      std::vector<SWTPoint2d> & compDimensions,
                      std::vector<SWTPointPair2d > & compBB );

std::vector<Chain> makeChains( const cv::Mat& colorImage,
                 std::vector<std::vector<SWTPoint2d> > & components,
                 std::vector<Point2dFloat> & compCenters,
                 std::vector<float> & compMedians,
                 std::vector<SWTPoint2d> & compDimensions,
                 std::vector<SWTPointPair2d > & compBB);

}

#endif // TEXTDETECTION_H

