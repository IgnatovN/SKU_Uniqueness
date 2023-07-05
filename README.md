# SKU Uniqueness
Mini-project from Karpov.Courses studying

### Description
Recommendations, along with search, help the user navigate and form the Customer Journey Map - the user's path from product to product while using website or application. So it is desirible to make this user journey as interesting and diverse as possible (enhance User Experience), in order to  help the user find the right product and not to bother him with the same type of recommendations.

The purpose of the project is to create uniqueness metric for recommended items to make a final list that will be showed to a user as diverse as possible and wrap it into a web service.

### Completed tasks
- Implement function based on KNN to estimate uniqueness of each item in item embeddings group
- Implement enhanced function based on KDE to estimate uniqueness of each item in item embeddings group
- Implement gruop diversity metric based on KDE uniqueness function to calculate diversity of item group
- Wrap functions above to FastAPI web-service
