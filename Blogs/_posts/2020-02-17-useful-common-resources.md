---
layout: post
title: Usefull Common Resources/Tricks
description: >
  
#image: /assets/img/blog/steve-harvey.jpg
comment: true
---

0. [Useful Links on General Study](#useful-links-on-general-study)
0. [Github: Configuration of local machine to github remote](#github-configuration-of-local-machine-to-github-remote)
0. [GDrive](#gdrive)
0. [Useful links on Jekyll, Google AdSense, Markdown. ](#useful-links-on-jekyll-google-adsense-markdown)
0. [Useful links on Travel & Insurance.  ](#useful-links-on-travel--insurance)
{:.message}


### Replace string in files recursively
* Simplest way to replace (all files, directory, recursive): 
    ``` 
    find . -type f -not -path '*/\.*' -exec sed -i 's/Previous string/New string/g' {} + 
    ```
* Note: Sometimes you might need to ignore some hidden files i.e. .git, you can use above command.
If you want to include hidden files use,
```
find . -type f  -exec sed -i 's/Previous string/New string/g' {} +
```
{:.message}


### Useful Links on General Study
* Information about probabilistic models of cognition
  * [Tom's Bayesian reading list](http://cocosci.princeton.edu/tom/bayes.html)

...[http://cocosci.princeton.edu/resources.php](http://cocosci.princeton.edu/resources.php)
{:.message}


### Useful links on Jekyll, Google AdSense, Markdown. 
* [Add Google AdSense to a Jekyll website](https://mycyberuniverse.com/en-gb/add-google-adsense-to-a-jekyll-website.html)
* [AdSense Jekyll + Github](http://www.lewisgavin.co.uk/Google-Analytics-Adsense/)

* [Jekyll Variables](https://jekyllrb.com/docs/variables/)

* [Add Google Analytics](https://michaelsoolee.com/google-analytics-jekyll/)
{:.message}



### Useful links on Travel & Insurance. 
* [Collecting Avios](https://www.britishairways.com/en-gb/executive-club/collecting-avios)

* [Virgin: What's covered](https://uk.virginmoney.com/virgin/travel-insurance/whats-covered.jsp)

* [AIG](https://www.aig.com.cn/individuals/travel-insurance?utm_source=baidu&utm_campaign=%E9%80%9A%E7%94%A8%E8%AF%8D%2D%E6%97%85%E8%A1%8C%E9%99%A9&utm_adgroup=%E9%80%9A%E7%94%A8%2D%E6%97%85%E6%B8%B8%E9%99%A9&utm_term=%E6%97%85%E8%A1%8C%E4%BF%9D%E9%99%A9&utm_medium=search%5Fcpc&utm_channel=baidu%5Fpc&utm_content=tyc&bd_vid=10708153713933454488)

* [京东安联](https://www.allianz360.com/?media=d4cd86c5e9444316994e5a2c00fa9cd6&type=1)

* [Aer Lingus: Dublin T2 => London Heathrow](https://www.aerlingus.com/html/flightSearchResult.html#/fareType=ONEWAY&fareCategory=ECONOMY&promoCode=&numAdults=1&numChildren=0&numInfants=0&groupBooking=false&sourceAirportCode_0=DUB&destinationAirportCode_0=LHR&departureDate_0=2019-10-19&flightCode_0=EI168)
{:.message}


### GDrive
* [GDrive Github](https://github.com/gdrive-org/gdrive)

* [List Folder, Root](https://github.com/gdrive-org/gdrive/issues/116)

* [Download entire folder](https://askubuntu.com/questions/867284/using-gdrive-to-download-entire-folder)

* [How to use GDrive in linux?](http://www.linuxandubuntu.com/home/google-drive-cli-client-for-linux)
{:.message}


### Github: Configuration of local machine to github remote
* [Local: Generating a new ssh key](https://help.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key)


* [Local: Adding your new SSH key to the ssh-agent](https://help.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#adding-your-ssh-key-to-the-ssh-agent)


* [Remote: Adding your new SSH key to your GitHub account](https://help.github.com/en/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account)


* [Common commands](https://github.com/XinshaoAmosWang/Deep-Metric-Embedding/blob/master/common_git.md)
{:.message}
