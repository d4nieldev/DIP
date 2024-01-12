## Explanation
[Distance transform](https://en.wikipedia.org/wiki/Distance_transform), also known as distance map or distance field, provides a metric or measure for the
distance of a pixel from the boundary of a component in an image where a component is defined as a set of connected pixels. We often use a signed distance field to
distinguish whether the pixel is inside or outside of the component.

We adopt the terms outer and outer distance transform to measure the distance field
inside and outside a component. The signed distance transform provides a combined measure for outer and
inner, where the inside region is positive, and the outside region is negative.


## How to use
The program takes 4 parameters:
1. an input image
2. component size threshold
3. distance type
4. optional output file.

Let us adopt the letters I, O, and S to indicate inner, outer, or signed distance
transform. The following example should compute the outer distance transform for components with
width and height above 15 pixels:

`python DistanceTransform.py input.jpg 15 O `

## Examples
`python DistanceTransform.py examples/test.png 15 I`

![image](https://github.com/d4nieldev/DistanceTransform/assets/72974081/76e32a1a-c699-4f21-8cb2-56078f94d957)

`python DistanceTransform.py examples/test.png 15 O`

![image](https://github.com/d4nieldev/DistanceTransform/assets/72974081/8f95df3f-ebb3-4488-9117-1e62b8ad774e)

`python DistanceTransform.py examples/test.png 15 S`

![image](https://github.com/d4nieldev/DistanceTransform/assets/72974081/fbdabafc-9d44-4de2-a601-2e9d81abe21a)
