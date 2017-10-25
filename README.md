# Data-Analysis-Project
Performing data analysis on 2 datasets in Python, to solve a fictional business problem 
Business problem: A data scientist wants to predict the probability of a user having his/her first engagement (i.e. a newsletter subscription) on the company's website. 

To solve this business problem, we can check pageviews and see if a higher number of pageviews correlates to a higher number of first engagements. 

So we load 2 datasets: session data and engagement data, which comes from two different systems. 
We can match users from both datasets by user_id. 

Steps of project: 
Load data into Python using pandas
Filter unneeded data out 
Combine dataFrames on user_id
Remove sessions preceeding the first engagement
Insert dependent variable y (engagement conversion)
Insert independent variable X (cumulative sum of pageviews)
Add logistric regression (using StatsModels)
Use matplotlib/seaborn to visualize results

Conclusion: Test hypothesis (higher pageviews means higher probability of first time user engagement)
Did this hold? 
