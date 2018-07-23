## Traffic Sign Classifier using Convolutional Neural Networks
This project consists of classifying (German) traffic signs using a convolutional neural network (CNN). A write-up is also available at [www.lrgonzales.com/traffic-sign-classifier](http://www.lrgonzales.com/traffic-sign-classifier).

### Introduction
Classifying street signs is a challenging and important real-world problem, particularly with the promise of self-driving cars. The environment in which classification takes place is relatively constrained in that street signs are typically standardized for a given geographical region and the camera/s used to "see" the traffic signs is/are assumed to be stationary with respect to an observant vehicle and positioned upright. However, varied lighting and weather conditions — and even blur due to velocity — are expected.

### Dataset
<p align="center">
  <img src="https://user-images.githubusercontent.com/4633154/38846095-7fe42122-41c8-11e8-9755-01c13528094b.jpg" width="480px" height="270px"/>
</p>
The above figure shows a sampling of the dataset used, the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news), such that each row corresponds to a unique class. Each sample is shown in the actual resolution used by the CNN — 32 x 32. Note that the dataset includes slight rotations, blurring, differing levels of brightness, and even glare within each class. Given the varied representations built into the dataset, data augmentation was not implemented. If data augmentation were to be considered, note that the oft-used vertical flip would be detrimental to some classes (e.g., last two rows of Fig. 1).

There are a total of 43 different classes. Below is a histogram of the classes in the training, validation, and test sets. The association between traffic sign name to label number can be found here [here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news).

<svg height="155pt" version="1.1" viewBox="0 0 213 155" width="213pt" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
 <defs>
  <style type="text/css">
*{stroke-linecap:butt;stroke-linejoin:round;}
  </style>
 </defs>
 <g id="figure_1">
  <g id="patch_1">
   <path d="M 0 155.748906 
L 213.555 155.748906 
L 213.555 0 
L 0 0 
z
" style="fill:none;"/>
  </g>
  <g id="axes_1">
   <g id="patch_2">
    <path d="M 49.405 43.311765 
L 202.855 43.311765 
L 202.855 10.7 
L 49.405 10.7 
z
" style="fill:none;"/>
   </g>
   <g id="matplotlib.axis_1">
    <g id="xtick_1">
     <g id="line2d_1">
      <defs>
       <path d="M 0 0 
L 0 3.5 
" id="mf10ce9e235" style="stroke:#000000;stroke-width:0.8;"/>
      </defs>
      <g>
       <use style="stroke:#000000;stroke-width:0.8;" x="49.405" xlink:href="#mf10ce9e235" y="43.311765"/>
      </g>
     </g>
    </g>
    <g id="xtick_2">
     <g id="line2d_2">
      <g>
       <use style="stroke:#000000;stroke-width:0.8;" x="85.940714" xlink:href="#mf10ce9e235" y="43.311765"/>
      </g>
     </g>
    </g>
    <g id="xtick_3">
     <g id="line2d_3">
      <g>
       <use style="stroke:#000000;stroke-width:0.8;" x="122.476429" xlink:href="#mf10ce9e235" y="43.311765"/>
      </g>
     </g>
    </g>
    <g id="xtick_4">
     <g id="line2d_4">
      <g>
       <use style="stroke:#000000;stroke-width:0.8;" x="159.012143" xlink:href="#mf10ce9e235" y="43.311765"/>
      </g>
     </g>
    </g>
    <g id="xtick_5">
     <g id="line2d_5">
      <g>
       <use style="stroke:#000000;stroke-width:0.8;" x="195.547857" xlink:href="#mf10ce9e235" y="43.311765"/>
      </g>
     </g>
    </g>
   </g>
   <g id="matplotlib.axis_2">
    <g id="ytick_1">
     <g id="line2d_6">
      <path clip-path="url(#pb77c48ae92)" d="M 49.405 43.311765 
L 202.855 43.311765 
" style="fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;"/>
     </g>
     <g id="line2d_7">
      <defs>
       <path d="M 0 0 
L -3.5 0 
" id="m5d0bc0d60a" style="stroke:#000000;stroke-width:0.8;"/>
      </defs>
      <g>
       <use style="stroke:#000000;stroke-width:0.8;" x="49.405" xlink:href="#m5d0bc0d60a" y="43.311765"/>
      </g>
     </g>
    </g>
    <g id="ytick_2">
     <g id="line2d_8">
      <path clip-path="url(#pb77c48ae92)" d="M 49.405 27.782353 
L 202.855 27.782353 
" style="fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;"/>
     </g>
     <g id="line2d_9">
      <g>
       <use style="stroke:#000000;stroke-width:0.8;" x="49.405" xlink:href="#m5d0bc0d60a" y="27.782353"/>
      </g>
     </g>
     <g id="text_1">
      <!-- 1000 -->
      <defs>
       <path d="M 12.40625 8.296875 
L 28.515625 8.296875 
L 28.515625 63.921875 
L 10.984375 60.40625 
L 10.984375 69.390625 
L 28.421875 72.90625 
L 38.28125 72.90625 
L 38.28125 8.296875 
L 54.390625 8.296875 
L 54.390625 0 
L 12.40625 0 
z
" id="DejaVuSans-31"/>
       <path d="M 31.78125 66.40625 
Q 24.171875 66.40625 20.328125 58.90625 
Q 16.5 51.421875 16.5 36.375 
Q 16.5 21.390625 20.328125 13.890625 
Q 24.171875 6.390625 31.78125 6.390625 
Q 39.453125 6.390625 43.28125 13.890625 
Q 47.125 21.390625 47.125 36.375 
Q 47.125 51.421875 43.28125 58.90625 
Q 39.453125 66.40625 31.78125 66.40625 
z
M 31.78125 74.21875 
Q 44.046875 74.21875 50.515625 64.515625 
Q 56.984375 54.828125 56.984375 36.375 
Q 56.984375 17.96875 50.515625 8.265625 
Q 44.046875 -1.421875 31.78125 -1.421875 
Q 19.53125 -1.421875 13.0625 8.265625 
Q 6.59375 17.96875 6.59375 36.375 
Q 6.59375 54.828125 13.0625 64.515625 
Q 19.53125 74.21875 31.78125 74.21875 
z
" id="DejaVuSans-30"/>
      </defs>
      <g transform="translate(20.7725 31.011689)scale(0.085 -0.085)">
       <use xlink:href="#DejaVuSans-31"/>
       <use x="63.623047" xlink:href="#DejaVuSans-30"/>
       <use x="127.246094" xlink:href="#DejaVuSans-30"/>
       <use x="190.869141" xlink:href="#DejaVuSans-30"/>
      </g>
     </g>
    </g>
    <g id="ytick_3">
     <g id="line2d_10">
      <path clip-path="url(#pb77c48ae92)" d="M 49.405 12.252941 
L 202.855 12.252941 
" style="fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;"/>
     </g>
     <g id="line2d_11">
      <g>
       <use style="stroke:#000000;stroke-width:0.8;" x="49.405" xlink:href="#m5d0bc0d60a" y="12.252941"/>
      </g>
     </g>
     <g id="text_2">
      <!-- 2000 -->
      <defs>
       <path d="M 19.1875 8.296875 
L 53.609375 8.296875 
L 53.609375 0 
L 7.328125 0 
L 7.328125 8.296875 
Q 12.9375 14.109375 22.625 23.890625 
Q 32.328125 33.6875 34.8125 36.53125 
Q 39.546875 41.84375 41.421875 45.53125 
Q 43.3125 49.21875 43.3125 52.78125 
Q 43.3125 58.59375 39.234375 62.25 
Q 35.15625 65.921875 28.609375 65.921875 
Q 23.96875 65.921875 18.8125 64.3125 
Q 13.671875 62.703125 7.8125 59.421875 
L 7.8125 69.390625 
Q 13.765625 71.78125 18.9375 73 
Q 24.125 74.21875 28.421875 74.21875 
Q 39.75 74.21875 46.484375 68.546875 
Q 53.21875 62.890625 53.21875 53.421875 
Q 53.21875 48.921875 51.53125 44.890625 
Q 49.859375 40.875 45.40625 35.40625 
Q 44.1875 33.984375 37.640625 27.21875 
Q 31.109375 20.453125 19.1875 8.296875 
z
" id="DejaVuSans-32"/>
      </defs>
      <g transform="translate(20.7725 15.482277)scale(0.085 -0.085)">
       <use xlink:href="#DejaVuSans-32"/>
       <use x="63.623047" xlink:href="#DejaVuSans-30"/>
       <use x="127.246094" xlink:href="#DejaVuSans-30"/>
       <use x="190.869141" xlink:href="#DejaVuSans-30"/>
      </g>
     </g>
    </g>
    <g id="ytick_4">
     <g id="line2d_12">
      <path clip-path="url(#pb77c48ae92)" d="M 49.405 43.311765 
L 202.855 43.311765 
" style="fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;"/>
     </g>
     <g id="line2d_13">
      <defs>
       <path d="M 0 0 
L -2 0 
" id="m9c092ab30a" style="stroke:#000000;stroke-width:0.6;"/>
      </defs>
      <g>
       <use style="stroke:#000000;stroke-width:0.6;" x="49.405" xlink:href="#m9c092ab30a" y="43.311765"/>
      </g>
     </g>
    </g>
    <g id="ytick_5">
     <g id="line2d_14">
      <path clip-path="url(#pb77c48ae92)" d="M 49.405 35.547059 
L 202.855 35.547059 
" style="fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;"/>
     </g>
     <g id="line2d_15">
      <g>
       <use style="stroke:#000000;stroke-width:0.6;" x="49.405" xlink:href="#m9c092ab30a" y="35.547059"/>
      </g>
     </g>
    </g>
    <g id="ytick_6">
     <g id="line2d_16">
      <path clip-path="url(#pb77c48ae92)" d="M 49.405 20.017647 
L 202.855 20.017647 
" style="fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;"/>
     </g>
     <g id="line2d_17">
      <g>
       <use style="stroke:#000000;stroke-width:0.6;" x="49.405" xlink:href="#m9c092ab30a" y="20.017647"/>
      </g>
     </g>
    </g>
    <g id="text_3">
     <!-- (a) -->
     <defs>
      <path d="M 31 75.875 
Q 24.46875 64.65625 21.28125 53.65625 
Q 18.109375 42.671875 18.109375 31.390625 
Q 18.109375 20.125 21.3125 9.0625 
Q 24.515625 -2 31 -13.1875 
L 23.1875 -13.1875 
Q 15.875 -1.703125 12.234375 9.375 
Q 8.59375 20.453125 8.59375 31.390625 
Q 8.59375 42.28125 12.203125 53.3125 
Q 15.828125 64.359375 23.1875 75.875 
z
" id="DejaVuSans-28"/>
      <path d="M 34.28125 27.484375 
Q 23.390625 27.484375 19.1875 25 
Q 14.984375 22.515625 14.984375 16.5 
Q 14.984375 11.71875 18.140625 8.90625 
Q 21.296875 6.109375 26.703125 6.109375 
Q 34.1875 6.109375 38.703125 11.40625 
Q 43.21875 16.703125 43.21875 25.484375 
L 43.21875 27.484375 
z
M 52.203125 31.203125 
L 52.203125 0 
L 43.21875 0 
L 43.21875 8.296875 
Q 40.140625 3.328125 35.546875 0.953125 
Q 30.953125 -1.421875 24.3125 -1.421875 
Q 15.921875 -1.421875 10.953125 3.296875 
Q 6 8.015625 6 15.921875 
Q 6 25.140625 12.171875 29.828125 
Q 18.359375 34.515625 30.609375 34.515625 
L 43.21875 34.515625 
L 43.21875 35.40625 
Q 43.21875 41.609375 39.140625 45 
Q 35.0625 48.390625 27.6875 48.390625 
Q 23 48.390625 18.546875 47.265625 
Q 14.109375 46.140625 10.015625 43.890625 
L 10.015625 52.203125 
Q 14.9375 54.109375 19.578125 55.046875 
Q 24.21875 56 28.609375 56 
Q 40.484375 56 46.34375 49.84375 
Q 52.203125 43.703125 52.203125 31.203125 
z
" id="DejaVuSans-61"/>
      <path d="M 8.015625 75.875 
L 15.828125 75.875 
Q 23.140625 64.359375 26.78125 53.3125 
Q 30.421875 42.28125 30.421875 31.390625 
Q 30.421875 20.453125 26.78125 9.375 
Q 23.140625 -1.703125 15.828125 -13.1875 
L 8.015625 -13.1875 
Q 14.5 -2 17.703125 9.0625 
Q 20.90625 20.125 20.90625 31.390625 
Q 20.90625 42.671875 17.703125 53.65625 
Q 14.5 64.65625 8.015625 75.875 
z
" id="DejaVuSans-29"/>
     </defs>
     <g transform="translate(7.2 25.342132)scale(0.08 -0.08)">
      <use xlink:href="#DejaVuSans-28"/>
      <use x="39.013672" xlink:href="#DejaVuSans-61"/>
      <use x="100.292969" xlink:href="#DejaVuSans-29"/>
     </g>
    </g>
   </g>
   <g id="patch_3">
    <path clip-path="url(#pb77c48ae92)" d="M 49.405 43.311765 
L 52.973605 43.311765 
L 52.973605 40.516471 
L 49.405 40.516471 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_4">
    <path clip-path="url(#pb77c48ae92)" d="M 52.973605 43.311765 
L 56.542209 43.311765 
L 56.542209 12.563529 
L 52.973605 12.563529 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_5">
    <path clip-path="url(#pb77c48ae92)" d="M 56.542209 43.311765 
L 60.110814 43.311765 
L 60.110814 12.097647 
L 56.542209 12.097647 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_6">
    <path clip-path="url(#pb77c48ae92)" d="M 60.110814 43.311765 
L 63.679419 43.311765 
L 63.679419 23.744706 
L 60.110814 23.744706 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_7">
    <path clip-path="url(#pb77c48ae92)" d="M 63.679419 43.311765 
L 67.248023 43.311765 
L 67.248023 15.824706 
L 63.679419 15.824706 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_8">
    <path clip-path="url(#pb77c48ae92)" d="M 67.248023 43.311765 
L 70.816628 43.311765 
L 70.816628 17.688235 
L 67.248023 17.688235 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_9">
    <path clip-path="url(#pb77c48ae92)" d="M 70.816628 43.311765 
L 74.385233 43.311765 
L 74.385233 37.721176 
L 70.816628 37.721176 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_10">
    <path clip-path="url(#pb77c48ae92)" d="M 74.385233 43.311765 
L 77.953837 43.311765 
L 77.953837 23.278824 
L 74.385233 23.278824 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_11">
    <path clip-path="url(#pb77c48ae92)" d="M 77.953837 43.311765 
L 81.522442 43.311765 
L 81.522442 23.744706 
L 77.953837 23.744706 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_12">
    <path clip-path="url(#pb77c48ae92)" d="M 81.522442 43.311765 
L 85.091047 43.311765 
L 85.091047 22.812941 
L 81.522442 22.812941 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_13">
    <path clip-path="url(#pb77c48ae92)" d="M 85.091047 43.311765 
L 88.659651 43.311765 
L 88.659651 15.358824 
L 85.091047 15.358824 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_14">
    <path clip-path="url(#pb77c48ae92)" d="M 88.659651 43.311765 
L 92.228256 43.311765 
L 92.228256 25.142353 
L 88.659651 25.142353 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_15">
    <path clip-path="url(#pb77c48ae92)" d="M 92.228256 43.311765 
L 95.79686 43.311765 
L 95.79686 13.961176 
L 92.228256 13.961176 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_16">
    <path clip-path="url(#pb77c48ae92)" d="M 95.79686 43.311765 
L 99.365465 43.311765 
L 99.365465 13.495294 
L 95.79686 13.495294 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_17">
    <path clip-path="url(#pb77c48ae92)" d="M 99.365465 43.311765 
L 102.93407 43.311765 
L 102.93407 32.596471 
L 99.365465 32.596471 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_18">
    <path clip-path="url(#pb77c48ae92)" d="M 102.93407 43.311765 
L 106.502674 43.311765 
L 106.502674 34.925882 
L 102.93407 34.925882 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_19">
    <path clip-path="url(#pb77c48ae92)" d="M 106.502674 43.311765 
L 110.071279 43.311765 
L 110.071279 37.721176 
L 106.502674 37.721176 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_20">
    <path clip-path="url(#pb77c48ae92)" d="M 110.071279 43.311765 
L 113.639884 43.311765 
L 113.639884 27.937647 
L 110.071279 27.937647 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_21">
    <path clip-path="url(#pb77c48ae92)" d="M 113.639884 43.311765 
L 117.208488 43.311765 
L 117.208488 26.54 
L 113.639884 26.54 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_22">
    <path clip-path="url(#pb77c48ae92)" d="M 117.208488 43.311765 
L 120.777093 43.311765 
L 120.777093 40.516471 
L 117.208488 40.516471 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_23">
    <path clip-path="url(#pb77c48ae92)" d="M 120.777093 43.311765 
L 124.345698 43.311765 
L 124.345698 38.652941 
L 120.777093 38.652941 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_24">
    <path clip-path="url(#pb77c48ae92)" d="M 124.345698 43.311765 
L 127.914302 43.311765 
L 127.914302 39.118824 
L 124.345698 39.118824 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_25">
    <path clip-path="url(#pb77c48ae92)" d="M 127.914302 43.311765 
L 131.482907 43.311765 
L 131.482907 38.187059 
L 127.914302 38.187059 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_26">
    <path clip-path="url(#pb77c48ae92)" d="M 131.482907 43.311765 
L 135.051512 43.311765 
L 135.051512 36.323529 
L 131.482907 36.323529 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_27">
    <path clip-path="url(#pb77c48ae92)" d="M 135.051512 43.311765 
L 138.620116 43.311765 
L 138.620116 39.584706 
L 135.051512 39.584706 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_28">
    <path clip-path="url(#pb77c48ae92)" d="M 138.620116 43.311765 
L 142.188721 43.311765 
L 142.188721 22.347059 
L 138.620116 22.347059 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_29">
    <path clip-path="url(#pb77c48ae92)" d="M 142.188721 43.311765 
L 145.757326 43.311765 
L 145.757326 34.925882 
L 142.188721 34.925882 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_30">
    <path clip-path="url(#pb77c48ae92)" d="M 145.757326 43.311765 
L 149.32593 43.311765 
L 149.32593 40.050588 
L 145.757326 40.050588 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_31">
    <path clip-path="url(#pb77c48ae92)" d="M 149.32593 43.311765 
L 152.894535 43.311765 
L 152.894535 35.857647 
L 149.32593 35.857647 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_32">
    <path clip-path="url(#pb77c48ae92)" d="M 152.894535 43.311765 
L 156.46314 43.311765 
L 156.46314 39.584706 
L 152.894535 39.584706 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_33">
    <path clip-path="url(#pb77c48ae92)" d="M 156.46314 43.311765 
L 160.031744 43.311765 
L 160.031744 37.255294 
L 156.46314 37.255294 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_34">
    <path clip-path="url(#pb77c48ae92)" d="M 160.031744 43.311765 
L 163.600349 43.311765 
L 163.600349 32.596471 
L 160.031744 32.596471 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_35">
    <path clip-path="url(#pb77c48ae92)" d="M 163.600349 43.311765 
L 167.168953 43.311765 
L 167.168953 40.050588 
L 163.600349 40.050588 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_36">
    <path clip-path="url(#pb77c48ae92)" d="M 167.168953 43.311765 
L 170.737558 43.311765 
L 170.737558 34.009647 
L 167.168953 34.009647 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_37">
    <path clip-path="url(#pb77c48ae92)" d="M 170.737558 43.311765 
L 174.306163 43.311765 
L 174.306163 37.721176 
L 170.737558 37.721176 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_38">
    <path clip-path="url(#pb77c48ae92)" d="M 174.306163 43.311765 
L 177.874767 43.311765 
L 177.874767 26.54 
L 174.306163 26.54 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_39">
    <path clip-path="url(#pb77c48ae92)" d="M 177.874767 43.311765 
L 181.443372 43.311765 
L 181.443372 38.187059 
L 177.874767 38.187059 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_40">
    <path clip-path="url(#pb77c48ae92)" d="M 181.443372 43.311765 
L 185.011977 43.311765 
L 185.011977 40.516471 
L 181.443372 40.516471 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_41">
    <path clip-path="url(#pb77c48ae92)" d="M 185.011977 43.311765 
L 188.580581 43.311765 
L 188.580581 14.427059 
L 185.011977 14.427059 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_42">
    <path clip-path="url(#pb77c48ae92)" d="M 188.580581 43.311765 
L 192.149186 43.311765 
L 192.149186 39.118824 
L 188.580581 39.118824 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_43">
    <path clip-path="url(#pb77c48ae92)" d="M 192.149186 43.311765 
L 195.717791 43.311765 
L 195.717791 38.652941 
L 192.149186 38.652941 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_44">
    <path clip-path="url(#pb77c48ae92)" d="M 195.717791 43.311765 
L 199.286395 43.311765 
L 199.286395 40.050588 
L 195.717791 40.050588 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_45">
    <path clip-path="url(#pb77c48ae92)" d="M 199.286395 43.311765 
L 202.855 43.311765 
L 202.855 40.050588 
L 199.286395 40.050588 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_46">
    <path d="M 49.405 43.311765 
L 49.405 10.7 
" style="fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;"/>
   </g>
   <g id="patch_47">
    <path d="M 49.405 43.311765 
L 202.855 43.311765 
" style="fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;"/>
   </g>
  </g>
  <g id="axes_2">
   <g id="patch_48">
    <path d="M 49.405 82.445882 
L 202.855 82.445882 
L 202.855 49.834118 
L 49.405 49.834118 
z
" style="fill:none;"/>
   </g>
   <g id="matplotlib.axis_3">
    <g id="xtick_6">
     <g id="line2d_18">
      <g>
       <use style="stroke:#000000;stroke-width:0.8;" x="49.405" xlink:href="#mf10ce9e235" y="82.445882"/>
      </g>
     </g>
    </g>
    <g id="xtick_7">
     <g id="line2d_19">
      <g>
       <use style="stroke:#000000;stroke-width:0.8;" x="85.940714" xlink:href="#mf10ce9e235" y="82.445882"/>
      </g>
     </g>
    </g>
    <g id="xtick_8">
     <g id="line2d_20">
      <g>
       <use style="stroke:#000000;stroke-width:0.8;" x="122.476429" xlink:href="#mf10ce9e235" y="82.445882"/>
      </g>
     </g>
    </g>
    <g id="xtick_9">
     <g id="line2d_21">
      <g>
       <use style="stroke:#000000;stroke-width:0.8;" x="159.012143" xlink:href="#mf10ce9e235" y="82.445882"/>
      </g>
     </g>
    </g>
    <g id="xtick_10">
     <g id="line2d_22">
      <g>
       <use style="stroke:#000000;stroke-width:0.8;" x="195.547857" xlink:href="#mf10ce9e235" y="82.445882"/>
      </g>
     </g>
    </g>
   </g>
   <g id="matplotlib.axis_4">
    <g id="ytick_7">
     <g id="line2d_23">
      <path clip-path="url(#pb230bd7aa5)" d="M 49.405 82.445882 
L 202.855 82.445882 
" style="fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;"/>
     </g>
     <g id="line2d_24">
      <g>
       <use style="stroke:#000000;stroke-width:0.8;" x="49.405" xlink:href="#m5d0bc0d60a" y="82.445882"/>
      </g>
     </g>
    </g>
    <g id="ytick_8">
     <g id="line2d_25">
      <path clip-path="url(#pb230bd7aa5)" d="M 49.405 69.401176 
L 202.855 69.401176 
" style="fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;"/>
     </g>
     <g id="line2d_26">
      <g>
       <use style="stroke:#000000;stroke-width:0.8;" x="49.405" xlink:href="#m5d0bc0d60a" y="69.401176"/>
      </g>
     </g>
     <g id="text_4">
      <!-- 100 -->
      <g transform="translate(26.180625 72.630512)scale(0.085 -0.085)">
       <use xlink:href="#DejaVuSans-31"/>
       <use x="63.623047" xlink:href="#DejaVuSans-30"/>
       <use x="127.246094" xlink:href="#DejaVuSans-30"/>
      </g>
     </g>
    </g>
    <g id="ytick_9">
     <g id="line2d_27">
      <path clip-path="url(#pb230bd7aa5)" d="M 49.405 56.356471 
L 202.855 56.356471 
" style="fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;"/>
     </g>
     <g id="line2d_28">
      <g>
       <use style="stroke:#000000;stroke-width:0.8;" x="49.405" xlink:href="#m5d0bc0d60a" y="56.356471"/>
      </g>
     </g>
     <g id="text_5">
      <!-- 200 -->
      <g transform="translate(26.180625 59.585807)scale(0.085 -0.085)">
       <use xlink:href="#DejaVuSans-32"/>
       <use x="63.623047" xlink:href="#DejaVuSans-30"/>
       <use x="127.246094" xlink:href="#DejaVuSans-30"/>
      </g>
     </g>
    </g>
    <g id="ytick_10">
     <g id="line2d_29">
      <path clip-path="url(#pb230bd7aa5)" d="M 49.405 75.923529 
L 202.855 75.923529 
" style="fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;"/>
     </g>
     <g id="line2d_30">
      <g>
       <use style="stroke:#000000;stroke-width:0.6;" x="49.405" xlink:href="#m9c092ab30a" y="75.923529"/>
      </g>
     </g>
    </g>
    <g id="ytick_11">
     <g id="line2d_31">
      <path clip-path="url(#pb230bd7aa5)" d="M 49.405 62.878824 
L 202.855 62.878824 
" style="fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;"/>
     </g>
     <g id="line2d_32">
      <g>
       <use style="stroke:#000000;stroke-width:0.6;" x="49.405" xlink:href="#m9c092ab30a" y="62.878824"/>
      </g>
     </g>
    </g>
    <g id="ytick_12">
     <g id="line2d_33">
      <path clip-path="url(#pb230bd7aa5)" d="M 49.405 49.834118 
L 202.855 49.834118 
" style="fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;"/>
     </g>
     <g id="line2d_34">
      <g>
       <use style="stroke:#000000;stroke-width:0.6;" x="49.405" xlink:href="#m9c092ab30a" y="49.834118"/>
      </g>
     </g>
    </g>
    <g id="text_6">
     <!-- (b) -->
     <defs>
      <path d="M 48.6875 27.296875 
Q 48.6875 37.203125 44.609375 42.84375 
Q 40.53125 48.484375 33.40625 48.484375 
Q 26.265625 48.484375 22.1875 42.84375 
Q 18.109375 37.203125 18.109375 27.296875 
Q 18.109375 17.390625 22.1875 11.75 
Q 26.265625 6.109375 33.40625 6.109375 
Q 40.53125 6.109375 44.609375 11.75 
Q 48.6875 17.390625 48.6875 27.296875 
z
M 18.109375 46.390625 
Q 20.953125 51.265625 25.265625 53.625 
Q 29.59375 56 35.59375 56 
Q 45.5625 56 51.78125 48.09375 
Q 58.015625 40.1875 58.015625 27.296875 
Q 58.015625 14.40625 51.78125 6.484375 
Q 45.5625 -1.421875 35.59375 -1.421875 
Q 29.59375 -1.421875 25.265625 0.953125 
Q 20.953125 3.328125 18.109375 8.203125 
L 18.109375 0 
L 9.078125 0 
L 9.078125 75.984375 
L 18.109375 75.984375 
z
" id="DejaVuSans-62"/>
     </defs>
     <g transform="translate(7.32 64.47625)scale(0.08 -0.08)">
      <use xlink:href="#DejaVuSans-28"/>
      <use x="39.013672" xlink:href="#DejaVuSans-62"/>
      <use x="102.490234" xlink:href="#DejaVuSans-29"/>
     </g>
    </g>
   </g>
   <g id="patch_49">
    <path clip-path="url(#pb230bd7aa5)" d="M 49.405 82.445882 
L 52.973605 82.445882 
L 52.973605 78.532471 
L 49.405 78.532471 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_50">
    <path clip-path="url(#pb230bd7aa5)" d="M 52.973605 82.445882 
L 56.542209 82.445882 
L 56.542209 51.138588 
L 52.973605 51.138588 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_51">
    <path clip-path="url(#pb230bd7aa5)" d="M 56.542209 82.445882 
L 60.110814 82.445882 
L 60.110814 51.138588 
L 56.542209 51.138588 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_52">
    <path clip-path="url(#pb230bd7aa5)" d="M 60.110814 82.445882 
L 63.679419 82.445882 
L 63.679419 62.878824 
L 60.110814 62.878824 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_53">
    <path clip-path="url(#pb230bd7aa5)" d="M 63.679419 82.445882 
L 67.248023 82.445882 
L 67.248023 55.052 
L 63.679419 55.052 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_54">
    <path clip-path="url(#pb230bd7aa5)" d="M 67.248023 82.445882 
L 70.816628 82.445882 
L 70.816628 55.052 
L 67.248023 55.052 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_55">
    <path clip-path="url(#pb230bd7aa5)" d="M 70.816628 82.445882 
L 74.385233 82.445882 
L 74.385233 74.619059 
L 70.816628 74.619059 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_56">
    <path clip-path="url(#pb230bd7aa5)" d="M 74.385233 82.445882 
L 77.953837 82.445882 
L 77.953837 62.878824 
L 74.385233 62.878824 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_57">
    <path clip-path="url(#pb230bd7aa5)" d="M 77.953837 82.445882 
L 81.522442 82.445882 
L 81.522442 62.878824 
L 77.953837 62.878824 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_58">
    <path clip-path="url(#pb230bd7aa5)" d="M 81.522442 82.445882 
L 85.091047 82.445882 
L 85.091047 62.878824 
L 81.522442 62.878824 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_59">
    <path clip-path="url(#pb230bd7aa5)" d="M 85.091047 82.445882 
L 88.659651 82.445882 
L 88.659651 55.052 
L 85.091047 55.052 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_60">
    <path clip-path="url(#pb230bd7aa5)" d="M 88.659651 82.445882 
L 92.228256 82.445882 
L 92.228256 62.878824 
L 88.659651 62.878824 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_61">
    <path clip-path="url(#pb230bd7aa5)" d="M 92.228256 82.445882 
L 95.79686 82.445882 
L 95.79686 55.052 
L 92.228256 55.052 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_62">
    <path clip-path="url(#pb230bd7aa5)" d="M 95.79686 82.445882 
L 99.365465 82.445882 
L 99.365465 51.138588 
L 95.79686 51.138588 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_63">
    <path clip-path="url(#pb230bd7aa5)" d="M 99.365465 82.445882 
L 102.93407 82.445882 
L 102.93407 70.705647 
L 99.365465 70.705647 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_64">
    <path clip-path="url(#pb230bd7aa5)" d="M 102.93407 82.445882 
L 106.502674 82.445882 
L 106.502674 70.705647 
L 102.93407 70.705647 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_65">
    <path clip-path="url(#pb230bd7aa5)" d="M 106.502674 82.445882 
L 110.071279 82.445882 
L 110.071279 74.619059 
L 106.502674 74.619059 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_66">
    <path clip-path="url(#pb230bd7aa5)" d="M 110.071279 82.445882 
L 113.639884 82.445882 
L 113.639884 66.792235 
L 110.071279 66.792235 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_67">
    <path clip-path="url(#pb230bd7aa5)" d="M 113.639884 82.445882 
L 117.208488 82.445882 
L 117.208488 66.792235 
L 113.639884 66.792235 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_68">
    <path clip-path="url(#pb230bd7aa5)" d="M 117.208488 82.445882 
L 120.777093 82.445882 
L 120.777093 78.532471 
L 117.208488 78.532471 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_69">
    <path clip-path="url(#pb230bd7aa5)" d="M 120.777093 82.445882 
L 124.345698 82.445882 
L 124.345698 74.619059 
L 120.777093 74.619059 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_70">
    <path clip-path="url(#pb230bd7aa5)" d="M 124.345698 82.445882 
L 127.914302 82.445882 
L 127.914302 74.619059 
L 124.345698 74.619059 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_71">
    <path clip-path="url(#pb230bd7aa5)" d="M 127.914302 82.445882 
L 131.482907 82.445882 
L 131.482907 74.619059 
L 127.914302 74.619059 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_72">
    <path clip-path="url(#pb230bd7aa5)" d="M 131.482907 82.445882 
L 135.051512 82.445882 
L 135.051512 74.619059 
L 131.482907 74.619059 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_73">
    <path clip-path="url(#pb230bd7aa5)" d="M 135.051512 82.445882 
L 138.620116 82.445882 
L 138.620116 78.532471 
L 135.051512 78.532471 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_74">
    <path clip-path="url(#pb230bd7aa5)" d="M 138.620116 82.445882 
L 142.188721 82.445882 
L 142.188721 62.878824 
L 138.620116 62.878824 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_75">
    <path clip-path="url(#pb230bd7aa5)" d="M 142.188721 82.445882 
L 145.757326 82.445882 
L 145.757326 74.619059 
L 142.188721 74.619059 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_76">
    <path clip-path="url(#pb230bd7aa5)" d="M 145.757326 82.445882 
L 149.32593 82.445882 
L 149.32593 78.532471 
L 145.757326 78.532471 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_77">
    <path clip-path="url(#pb230bd7aa5)" d="M 149.32593 82.445882 
L 152.894535 82.445882 
L 152.894535 74.619059 
L 149.32593 74.619059 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_78">
    <path clip-path="url(#pb230bd7aa5)" d="M 152.894535 82.445882 
L 156.46314 82.445882 
L 156.46314 78.532471 
L 152.894535 78.532471 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_79">
    <path clip-path="url(#pb230bd7aa5)" d="M 156.46314 82.445882 
L 160.031744 82.445882 
L 160.031744 74.619059 
L 156.46314 74.619059 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_80">
    <path clip-path="url(#pb230bd7aa5)" d="M 160.031744 82.445882 
L 163.600349 82.445882 
L 163.600349 70.705647 
L 160.031744 70.705647 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_81">
    <path clip-path="url(#pb230bd7aa5)" d="M 163.600349 82.445882 
L 167.168953 82.445882 
L 167.168953 78.532471 
L 163.600349 78.532471 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_82">
    <path clip-path="url(#pb230bd7aa5)" d="M 167.168953 82.445882 
L 170.737558 82.445882 
L 170.737558 70.705647 
L 167.168953 70.705647 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_83">
    <path clip-path="url(#pb230bd7aa5)" d="M 170.737558 82.445882 
L 174.306163 82.445882 
L 174.306163 74.619059 
L 170.737558 74.619059 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_84">
    <path clip-path="url(#pb230bd7aa5)" d="M 174.306163 82.445882 
L 177.874767 82.445882 
L 177.874767 66.792235 
L 174.306163 66.792235 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_85">
    <path clip-path="url(#pb230bd7aa5)" d="M 177.874767 82.445882 
L 181.443372 82.445882 
L 181.443372 74.619059 
L 177.874767 74.619059 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_86">
    <path clip-path="url(#pb230bd7aa5)" d="M 181.443372 82.445882 
L 185.011977 82.445882 
L 185.011977 78.532471 
L 181.443372 78.532471 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_87">
    <path clip-path="url(#pb230bd7aa5)" d="M 185.011977 82.445882 
L 188.580581 82.445882 
L 188.580581 55.052 
L 185.011977 55.052 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_88">
    <path clip-path="url(#pb230bd7aa5)" d="M 188.580581 82.445882 
L 192.149186 82.445882 
L 192.149186 78.532471 
L 188.580581 78.532471 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_89">
    <path clip-path="url(#pb230bd7aa5)" d="M 192.149186 82.445882 
L 195.717791 82.445882 
L 195.717791 74.619059 
L 192.149186 74.619059 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_90">
    <path clip-path="url(#pb230bd7aa5)" d="M 195.717791 82.445882 
L 199.286395 82.445882 
L 199.286395 78.532471 
L 195.717791 78.532471 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_91">
    <path clip-path="url(#pb230bd7aa5)" d="M 199.286395 82.445882 
L 202.855 82.445882 
L 202.855 78.532471 
L 199.286395 78.532471 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_92">
    <path d="M 49.405 82.445882 
L 49.405 49.834118 
" style="fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;"/>
   </g>
   <g id="patch_93">
    <path d="M 49.405 82.445882 
L 202.855 82.445882 
" style="fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;"/>
   </g>
  </g>
  <g id="axes_3">
   <g id="patch_94">
    <path d="M 49.405 121.58 
L 202.855 121.58 
L 202.855 88.968235 
L 49.405 88.968235 
z
" style="fill:none;"/>
   </g>
   <g id="matplotlib.axis_5">
    <g id="xtick_11">
     <g id="line2d_35">
      <g>
       <use style="stroke:#000000;stroke-width:0.8;" x="49.405" xlink:href="#mf10ce9e235" y="121.58"/>
      </g>
     </g>
     <g id="text_7">
      <!-- 0 -->
      <g transform="translate(46.700938 135.038672)scale(0.085 -0.085)">
       <use xlink:href="#DejaVuSans-30"/>
      </g>
     </g>
    </g>
    <g id="xtick_12">
     <g id="line2d_36">
      <g>
       <use style="stroke:#000000;stroke-width:0.8;" x="85.940714" xlink:href="#mf10ce9e235" y="121.58"/>
      </g>
     </g>
     <g id="text_8">
      <!-- 10 -->
      <g transform="translate(80.532589 135.038672)scale(0.085 -0.085)">
       <use xlink:href="#DejaVuSans-31"/>
       <use x="63.623047" xlink:href="#DejaVuSans-30"/>
      </g>
     </g>
    </g>
    <g id="xtick_13">
     <g id="line2d_37">
      <g>
       <use style="stroke:#000000;stroke-width:0.8;" x="122.476429" xlink:href="#mf10ce9e235" y="121.58"/>
      </g>
     </g>
     <g id="text_9">
      <!-- 20 -->
      <g transform="translate(117.068304 135.038672)scale(0.085 -0.085)">
       <use xlink:href="#DejaVuSans-32"/>
       <use x="63.623047" xlink:href="#DejaVuSans-30"/>
      </g>
     </g>
    </g>
    <g id="xtick_14">
     <g id="line2d_38">
      <g>
       <use style="stroke:#000000;stroke-width:0.8;" x="159.012143" xlink:href="#mf10ce9e235" y="121.58"/>
      </g>
     </g>
     <g id="text_10">
      <!-- 30 -->
      <defs>
       <path d="M 40.578125 39.3125 
Q 47.65625 37.796875 51.625 33 
Q 55.609375 28.21875 55.609375 21.1875 
Q 55.609375 10.40625 48.1875 4.484375 
Q 40.765625 -1.421875 27.09375 -1.421875 
Q 22.515625 -1.421875 17.65625 -0.515625 
Q 12.796875 0.390625 7.625 2.203125 
L 7.625 11.71875 
Q 11.71875 9.328125 16.59375 8.109375 
Q 21.484375 6.890625 26.8125 6.890625 
Q 36.078125 6.890625 40.9375 10.546875 
Q 45.796875 14.203125 45.796875 21.1875 
Q 45.796875 27.640625 41.28125 31.265625 
Q 36.765625 34.90625 28.71875 34.90625 
L 20.21875 34.90625 
L 20.21875 43.015625 
L 29.109375 43.015625 
Q 36.375 43.015625 40.234375 45.921875 
Q 44.09375 48.828125 44.09375 54.296875 
Q 44.09375 59.90625 40.109375 62.90625 
Q 36.140625 65.921875 28.71875 65.921875 
Q 24.65625 65.921875 20.015625 65.03125 
Q 15.375 64.15625 9.8125 62.3125 
L 9.8125 71.09375 
Q 15.4375 72.65625 20.34375 73.4375 
Q 25.25 74.21875 29.59375 74.21875 
Q 40.828125 74.21875 47.359375 69.109375 
Q 53.90625 64.015625 53.90625 55.328125 
Q 53.90625 49.265625 50.4375 45.09375 
Q 46.96875 40.921875 40.578125 39.3125 
z
" id="DejaVuSans-33"/>
      </defs>
      <g transform="translate(153.604018 135.038672)scale(0.085 -0.085)">
       <use xlink:href="#DejaVuSans-33"/>
       <use x="63.623047" xlink:href="#DejaVuSans-30"/>
      </g>
     </g>
    </g>
    <g id="xtick_15">
     <g id="line2d_39">
      <g>
       <use style="stroke:#000000;stroke-width:0.8;" x="195.547857" xlink:href="#mf10ce9e235" y="121.58"/>
      </g>
     </g>
     <g id="text_11">
      <!-- 40 -->
      <defs>
       <path d="M 37.796875 64.3125 
L 12.890625 25.390625 
L 37.796875 25.390625 
z
M 35.203125 72.90625 
L 47.609375 72.90625 
L 47.609375 25.390625 
L 58.015625 25.390625 
L 58.015625 17.1875 
L 47.609375 17.1875 
L 47.609375 0 
L 37.796875 0 
L 37.796875 17.1875 
L 4.890625 17.1875 
L 4.890625 26.703125 
z
" id="DejaVuSans-34"/>
      </defs>
      <g transform="translate(190.139732 135.038672)scale(0.085 -0.085)">
       <use xlink:href="#DejaVuSans-34"/>
       <use x="63.623047" xlink:href="#DejaVuSans-30"/>
      </g>
     </g>
    </g>
    <g id="text_12">
     <!-- Class labels -->
     <defs>
      <path d="M 64.40625 67.28125 
L 64.40625 56.890625 
Q 59.421875 61.53125 53.78125 63.8125 
Q 48.140625 66.109375 41.796875 66.109375 
Q 29.296875 66.109375 22.65625 58.46875 
Q 16.015625 50.828125 16.015625 36.375 
Q 16.015625 21.96875 22.65625 14.328125 
Q 29.296875 6.6875 41.796875 6.6875 
Q 48.140625 6.6875 53.78125 8.984375 
Q 59.421875 11.28125 64.40625 15.921875 
L 64.40625 5.609375 
Q 59.234375 2.09375 53.4375 0.328125 
Q 47.65625 -1.421875 41.21875 -1.421875 
Q 24.65625 -1.421875 15.125 8.703125 
Q 5.609375 18.84375 5.609375 36.375 
Q 5.609375 53.953125 15.125 64.078125 
Q 24.65625 74.21875 41.21875 74.21875 
Q 47.75 74.21875 53.53125 72.484375 
Q 59.328125 70.75 64.40625 67.28125 
z
" id="DejaVuSans-43"/>
      <path d="M 9.421875 75.984375 
L 18.40625 75.984375 
L 18.40625 0 
L 9.421875 0 
z
" id="DejaVuSans-6c"/>
      <path d="M 44.28125 53.078125 
L 44.28125 44.578125 
Q 40.484375 46.53125 36.375 47.5 
Q 32.28125 48.484375 27.875 48.484375 
Q 21.1875 48.484375 17.84375 46.4375 
Q 14.5 44.390625 14.5 40.28125 
Q 14.5 37.15625 16.890625 35.375 
Q 19.28125 33.59375 26.515625 31.984375 
L 29.59375 31.296875 
Q 39.15625 29.25 43.1875 25.515625 
Q 47.21875 21.78125 47.21875 15.09375 
Q 47.21875 7.46875 41.1875 3.015625 
Q 35.15625 -1.421875 24.609375 -1.421875 
Q 20.21875 -1.421875 15.453125 -0.5625 
Q 10.6875 0.296875 5.421875 2 
L 5.421875 11.28125 
Q 10.40625 8.6875 15.234375 7.390625 
Q 20.0625 6.109375 24.8125 6.109375 
Q 31.15625 6.109375 34.5625 8.28125 
Q 37.984375 10.453125 37.984375 14.40625 
Q 37.984375 18.0625 35.515625 20.015625 
Q 33.0625 21.96875 24.703125 23.78125 
L 21.578125 24.515625 
Q 13.234375 26.265625 9.515625 29.90625 
Q 5.8125 33.546875 5.8125 39.890625 
Q 5.8125 47.609375 11.28125 51.796875 
Q 16.75 56 26.8125 56 
Q 31.78125 56 36.171875 55.265625 
Q 40.578125 54.546875 44.28125 53.078125 
z
" id="DejaVuSans-73"/>
      <path id="DejaVuSans-20"/>
      <path d="M 56.203125 29.59375 
L 56.203125 25.203125 
L 14.890625 25.203125 
Q 15.484375 15.921875 20.484375 11.0625 
Q 25.484375 6.203125 34.421875 6.203125 
Q 39.59375 6.203125 44.453125 7.46875 
Q 49.3125 8.734375 54.109375 11.28125 
L 54.109375 2.78125 
Q 49.265625 0.734375 44.1875 -0.34375 
Q 39.109375 -1.421875 33.890625 -1.421875 
Q 20.796875 -1.421875 13.15625 6.1875 
Q 5.515625 13.8125 5.515625 26.8125 
Q 5.515625 40.234375 12.765625 48.109375 
Q 20.015625 56 32.328125 56 
Q 43.359375 56 49.78125 48.890625 
Q 56.203125 41.796875 56.203125 29.59375 
z
M 47.21875 32.234375 
Q 47.125 39.59375 43.09375 43.984375 
Q 39.0625 48.390625 32.421875 48.390625 
Q 24.90625 48.390625 20.390625 44.140625 
Q 15.875 39.890625 15.1875 32.171875 
z
" id="DejaVuSans-65"/>
     </defs>
     <g transform="translate(102.5775 146.885156)scale(0.08 -0.08)">
      <use xlink:href="#DejaVuSans-43"/>
      <use x="69.824219" xlink:href="#DejaVuSans-6c"/>
      <use x="97.607422" xlink:href="#DejaVuSans-61"/>
      <use x="158.886719" xlink:href="#DejaVuSans-73"/>
      <use x="210.986328" xlink:href="#DejaVuSans-73"/>
      <use x="263.085938" xlink:href="#DejaVuSans-20"/>
      <use x="294.873047" xlink:href="#DejaVuSans-6c"/>
      <use x="322.65625" xlink:href="#DejaVuSans-61"/>
      <use x="383.935547" xlink:href="#DejaVuSans-62"/>
      <use x="447.412109" xlink:href="#DejaVuSans-65"/>
      <use x="508.935547" xlink:href="#DejaVuSans-6c"/>
      <use x="536.71875" xlink:href="#DejaVuSans-73"/>
     </g>
    </g>
   </g>
   <g id="matplotlib.axis_6">
    <g id="ytick_13">
     <g id="line2d_40">
      <path clip-path="url(#p36f9f48430)" d="M 49.405 121.58 
L 202.855 121.58 
" style="fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;"/>
     </g>
     <g id="line2d_41">
      <g>
       <use style="stroke:#000000;stroke-width:0.8;" x="49.405" xlink:href="#m5d0bc0d60a" y="121.58"/>
      </g>
     </g>
    </g>
    <g id="ytick_14">
     <g id="line2d_42">
      <path clip-path="url(#p36f9f48430)" d="M 49.405 109.350588 
L 202.855 109.350588 
" style="fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;"/>
     </g>
     <g id="line2d_43">
      <g>
       <use style="stroke:#000000;stroke-width:0.8;" x="49.405" xlink:href="#m5d0bc0d60a" y="109.350588"/>
      </g>
     </g>
     <g id="text_13">
      <!-- 300 -->
      <g transform="translate(26.180625 112.579924)scale(0.085 -0.085)">
       <use xlink:href="#DejaVuSans-33"/>
       <use x="63.623047" xlink:href="#DejaVuSans-30"/>
       <use x="127.246094" xlink:href="#DejaVuSans-30"/>
      </g>
     </g>
    </g>
    <g id="ytick_15">
     <g id="line2d_44">
      <path clip-path="url(#p36f9f48430)" d="M 49.405 97.121176 
L 202.855 97.121176 
" style="fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;"/>
     </g>
     <g id="line2d_45">
      <g>
       <use style="stroke:#000000;stroke-width:0.8;" x="49.405" xlink:href="#m5d0bc0d60a" y="97.121176"/>
      </g>
     </g>
     <g id="text_14">
      <!-- 600 -->
      <defs>
       <path d="M 33.015625 40.375 
Q 26.375 40.375 22.484375 35.828125 
Q 18.609375 31.296875 18.609375 23.390625 
Q 18.609375 15.53125 22.484375 10.953125 
Q 26.375 6.390625 33.015625 6.390625 
Q 39.65625 6.390625 43.53125 10.953125 
Q 47.40625 15.53125 47.40625 23.390625 
Q 47.40625 31.296875 43.53125 35.828125 
Q 39.65625 40.375 33.015625 40.375 
z
M 52.59375 71.296875 
L 52.59375 62.3125 
Q 48.875 64.0625 45.09375 64.984375 
Q 41.3125 65.921875 37.59375 65.921875 
Q 27.828125 65.921875 22.671875 59.328125 
Q 17.53125 52.734375 16.796875 39.40625 
Q 19.671875 43.65625 24.015625 45.921875 
Q 28.375 48.1875 33.59375 48.1875 
Q 44.578125 48.1875 50.953125 41.515625 
Q 57.328125 34.859375 57.328125 23.390625 
Q 57.328125 12.15625 50.6875 5.359375 
Q 44.046875 -1.421875 33.015625 -1.421875 
Q 20.359375 -1.421875 13.671875 8.265625 
Q 6.984375 17.96875 6.984375 36.375 
Q 6.984375 53.65625 15.1875 63.9375 
Q 23.390625 74.21875 37.203125 74.21875 
Q 40.921875 74.21875 44.703125 73.484375 
Q 48.484375 72.75 52.59375 71.296875 
z
" id="DejaVuSans-36"/>
      </defs>
      <g transform="translate(26.180625 100.350512)scale(0.085 -0.085)">
       <use xlink:href="#DejaVuSans-36"/>
       <use x="63.623047" xlink:href="#DejaVuSans-30"/>
       <use x="127.246094" xlink:href="#DejaVuSans-30"/>
      </g>
     </g>
    </g>
    <g id="ytick_16">
     <g id="line2d_46">
      <path clip-path="url(#p36f9f48430)" d="M 49.405 115.465294 
L 202.855 115.465294 
" style="fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;"/>
     </g>
     <g id="line2d_47">
      <g>
       <use style="stroke:#000000;stroke-width:0.6;" x="49.405" xlink:href="#m9c092ab30a" y="115.465294"/>
      </g>
     </g>
    </g>
    <g id="ytick_17">
     <g id="line2d_48">
      <path clip-path="url(#p36f9f48430)" d="M 49.405 103.235882 
L 202.855 103.235882 
" style="fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;"/>
     </g>
     <g id="line2d_49">
      <g>
       <use style="stroke:#000000;stroke-width:0.6;" x="49.405" xlink:href="#m9c092ab30a" y="103.235882"/>
      </g>
     </g>
    </g>
    <g id="ytick_18">
     <g id="line2d_50">
      <path clip-path="url(#p36f9f48430)" d="M 49.405 91.006471 
L 202.855 91.006471 
" style="fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;"/>
     </g>
     <g id="line2d_51">
      <g>
       <use style="stroke:#000000;stroke-width:0.6;" x="49.405" xlink:href="#m9c092ab30a" y="91.006471"/>
      </g>
     </g>
    </g>
    <g id="text_15">
     <!-- (c) -->
     <defs>
      <path d="M 48.78125 52.59375 
L 48.78125 44.1875 
Q 44.96875 46.296875 41.140625 47.34375 
Q 37.3125 48.390625 33.40625 48.390625 
Q 24.65625 48.390625 19.8125 42.84375 
Q 14.984375 37.3125 14.984375 27.296875 
Q 14.984375 17.28125 19.8125 11.734375 
Q 24.65625 6.203125 33.40625 6.203125 
Q 37.3125 6.203125 41.140625 7.25 
Q 44.96875 8.296875 48.78125 10.40625 
L 48.78125 2.09375 
Q 45.015625 0.34375 40.984375 -0.53125 
Q 36.96875 -1.421875 32.421875 -1.421875 
Q 20.0625 -1.421875 12.78125 6.34375 
Q 5.515625 14.109375 5.515625 27.296875 
Q 5.515625 40.671875 12.859375 48.328125 
Q 20.21875 56 33.015625 56 
Q 37.15625 56 41.109375 55.140625 
Q 45.0625 54.296875 48.78125 52.59375 
z
" id="DejaVuSans-63"/>
     </defs>
     <g transform="translate(7.56 103.610368)scale(0.08 -0.08)">
      <use xlink:href="#DejaVuSans-28"/>
      <use x="39.013672" xlink:href="#DejaVuSans-63"/>
      <use x="93.994141" xlink:href="#DejaVuSans-29"/>
     </g>
    </g>
   </g>
   <g id="patch_95">
    <path clip-path="url(#p36f9f48430)" d="M 49.405 121.58 
L 52.973605 121.58 
L 52.973605 119.134118 
L 49.405 119.134118 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_96">
    <path clip-path="url(#p36f9f48430)" d="M 52.973605 121.58 
L 56.542209 121.58 
L 56.542209 92.229412 
L 52.973605 92.229412 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_97">
    <path clip-path="url(#p36f9f48430)" d="M 56.542209 121.58 
L 60.110814 121.58 
L 60.110814 91.006471 
L 56.542209 91.006471 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_98">
    <path clip-path="url(#p36f9f48430)" d="M 60.110814 121.58 
L 63.679419 121.58 
L 63.679419 103.235882 
L 60.110814 103.235882 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_99">
    <path clip-path="url(#p36f9f48430)" d="M 63.679419 121.58 
L 67.248023 121.58 
L 67.248023 94.675294 
L 63.679419 94.675294 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_100">
    <path clip-path="url(#p36f9f48430)" d="M 67.248023 121.58 
L 70.816628 121.58 
L 70.816628 95.898235 
L 67.248023 95.898235 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_101">
    <path clip-path="url(#p36f9f48430)" d="M 70.816628 121.58 
L 74.385233 121.58 
L 74.385233 115.465294 
L 70.816628 115.465294 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_102">
    <path clip-path="url(#p36f9f48430)" d="M 74.385233 121.58 
L 77.953837 121.58 
L 77.953837 103.235882 
L 74.385233 103.235882 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_103">
    <path clip-path="url(#p36f9f48430)" d="M 77.953837 121.58 
L 81.522442 121.58 
L 81.522442 103.235882 
L 77.953837 103.235882 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_104">
    <path clip-path="url(#p36f9f48430)" d="M 81.522442 121.58 
L 85.091047 121.58 
L 85.091047 102.012941 
L 81.522442 102.012941 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_105">
    <path clip-path="url(#p36f9f48430)" d="M 85.091047 121.58 
L 88.659651 121.58 
L 88.659651 94.675294 
L 85.091047 94.675294 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_106">
    <path clip-path="url(#p36f9f48430)" d="M 88.659651 121.58 
L 92.228256 121.58 
L 92.228256 104.458824 
L 88.659651 104.458824 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_107">
    <path clip-path="url(#p36f9f48430)" d="M 92.228256 121.58 
L 95.79686 121.58 
L 95.79686 93.452353 
L 92.228256 93.452353 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_108">
    <path clip-path="url(#p36f9f48430)" d="M 95.79686 121.58 
L 99.365465 121.58 
L 99.365465 92.229412 
L 95.79686 92.229412 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_109">
    <path clip-path="url(#p36f9f48430)" d="M 99.365465 121.58 
L 102.93407 121.58 
L 102.93407 110.573529 
L 99.365465 110.573529 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_110">
    <path clip-path="url(#p36f9f48430)" d="M 102.93407 121.58 
L 106.502674 121.58 
L 106.502674 113.019412 
L 102.93407 113.019412 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_111">
    <path clip-path="url(#p36f9f48430)" d="M 106.502674 121.58 
L 110.071279 121.58 
L 110.071279 115.465294 
L 106.502674 115.465294 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_112">
    <path clip-path="url(#p36f9f48430)" d="M 110.071279 121.58 
L 113.639884 121.58 
L 113.639884 106.904706 
L 110.071279 106.904706 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_113">
    <path clip-path="url(#p36f9f48430)" d="M 113.639884 121.58 
L 117.208488 121.58 
L 117.208488 105.681765 
L 113.639884 105.681765 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_114">
    <path clip-path="url(#p36f9f48430)" d="M 117.208488 121.58 
L 120.777093 121.58 
L 120.777093 119.134118 
L 117.208488 119.134118 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_115">
    <path clip-path="url(#p36f9f48430)" d="M 120.777093 121.58 
L 124.345698 121.58 
L 124.345698 117.911176 
L 120.777093 117.911176 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_116">
    <path clip-path="url(#p36f9f48430)" d="M 124.345698 121.58 
L 127.914302 121.58 
L 127.914302 117.911176 
L 124.345698 117.911176 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_117">
    <path clip-path="url(#p36f9f48430)" d="M 127.914302 121.58 
L 131.482907 121.58 
L 131.482907 116.688235 
L 127.914302 116.688235 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_118">
    <path clip-path="url(#p36f9f48430)" d="M 131.482907 121.58 
L 135.051512 121.58 
L 135.051512 115.465294 
L 131.482907 115.465294 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_119">
    <path clip-path="url(#p36f9f48430)" d="M 135.051512 121.58 
L 138.620116 121.58 
L 138.620116 117.911176 
L 135.051512 117.911176 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_120">
    <path clip-path="url(#p36f9f48430)" d="M 138.620116 121.58 
L 142.188721 121.58 
L 142.188721 102.012941 
L 138.620116 102.012941 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_121">
    <path clip-path="url(#p36f9f48430)" d="M 142.188721 121.58 
L 145.757326 121.58 
L 145.757326 114.242353 
L 142.188721 114.242353 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_122">
    <path clip-path="url(#p36f9f48430)" d="M 145.757326 121.58 
L 149.32593 121.58 
L 149.32593 119.134118 
L 145.757326 119.134118 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_123">
    <path clip-path="url(#p36f9f48430)" d="M 149.32593 121.58 
L 152.894535 121.58 
L 152.894535 115.465294 
L 149.32593 115.465294 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_124">
    <path clip-path="url(#p36f9f48430)" d="M 152.894535 121.58 
L 156.46314 121.58 
L 156.46314 117.911176 
L 152.894535 117.911176 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_125">
    <path clip-path="url(#p36f9f48430)" d="M 156.46314 121.58 
L 160.031744 121.58 
L 160.031744 115.465294 
L 156.46314 115.465294 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_126">
    <path clip-path="url(#p36f9f48430)" d="M 160.031744 121.58 
L 163.600349 121.58 
L 163.600349 110.573529 
L 160.031744 110.573529 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_127">
    <path clip-path="url(#p36f9f48430)" d="M 163.600349 121.58 
L 167.168953 121.58 
L 167.168953 119.134118 
L 163.600349 119.134118 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_128">
    <path clip-path="url(#p36f9f48430)" d="M 167.168953 121.58 
L 170.737558 121.58 
L 170.737558 113.019412 
L 167.168953 113.019412 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_129">
    <path clip-path="url(#p36f9f48430)" d="M 170.737558 121.58 
L 174.306163 121.58 
L 174.306163 116.688235 
L 170.737558 116.688235 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_130">
    <path clip-path="url(#p36f9f48430)" d="M 174.306163 121.58 
L 177.874767 121.58 
L 177.874767 105.681765 
L 174.306163 105.681765 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_131">
    <path clip-path="url(#p36f9f48430)" d="M 177.874767 121.58 
L 181.443372 121.58 
L 181.443372 116.688235 
L 177.874767 116.688235 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_132">
    <path clip-path="url(#p36f9f48430)" d="M 181.443372 121.58 
L 185.011977 121.58 
L 185.011977 119.134118 
L 181.443372 119.134118 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_133">
    <path clip-path="url(#p36f9f48430)" d="M 185.011977 121.58 
L 188.580581 121.58 
L 188.580581 93.452353 
L 185.011977 93.452353 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_134">
    <path clip-path="url(#p36f9f48430)" d="M 188.580581 121.58 
L 192.149186 121.58 
L 192.149186 117.911176 
L 188.580581 117.911176 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_135">
    <path clip-path="url(#p36f9f48430)" d="M 192.149186 121.58 
L 195.717791 121.58 
L 195.717791 117.911176 
L 192.149186 117.911176 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_136">
    <path clip-path="url(#p36f9f48430)" d="M 195.717791 121.58 
L 199.286395 121.58 
L 199.286395 119.134118 
L 195.717791 119.134118 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_137">
    <path clip-path="url(#p36f9f48430)" d="M 199.286395 121.58 
L 202.855 121.58 
L 202.855 117.911176 
L 199.286395 117.911176 
z
" style="fill:#41cefa;stroke:#000000;stroke-linejoin:miter;"/>
   </g>
   <g id="patch_138">
    <path d="M 49.405 121.58 
L 49.405 88.968235 
" style="fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;"/>
   </g>
   <g id="patch_139">
    <path d="M 49.405 121.58 
L 202.855 121.58 
" style="fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;"/>
   </g>
  </g>
 </g>
 <defs>
  <clipPath id="pb77c48ae92">
   <rect height="32.611765" width="153.45" x="49.405" y="10.7"/>
  </clipPath>
  <clipPath id="pb230bd7aa5">
   <rect height="32.611765" width="153.45" x="49.405" y="49.834118"/>
  </clipPath>
  <clipPath id="p36f9f48430">
   <rect height="32.611765" width="153.45" x="49.405" y="88.968235"/>
  </clipPath>
 </defs>
</svg>

Preprocessing consists of resizing all of the images to a dimension of 32 x 32 x 3 (RGB), converting to grayscale [2], and normalizing from [0,255] to [−1,1).

### CNN Architecture and Training


### Improvements
Throughout the design, it became apparent that the methods used rely on a fairly smooth road, free of obstacles and relatively free of debris. For this reason, the input was preprocessed with a Gaussian blur. However, a Gaussian blur may not be enough for some unusual road surfaces or when obstacles or large debris are present. The danger is that these features could be mistaken for a lane in the algorithm. More advanced techniques, such as convolutional neural networks, could be more robust to such non-idealities.

The `src/hough.py` implementation of the Hough transform could benefit from an accumulator implemented with more efficient storage than a `list`. Doing so would likely result in on-par performance to the OpenCV implementation.

### Usage
The program can be performed on a `jpg` or `mp4` file from the `input` directory by executing `python lane_recog.py input/<file>` where `<file>` is the desired input.

`lane_recog.py` does not make use of `src/edge_det.py` or `src/hough.py`.

### Dependencies
`lane_recog.py` makes use of `numpy`, `matplotlib`, `cv2`, and `moviepy` (for `.mp4` input files).
