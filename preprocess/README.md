#### 数据预处理1

preprocess.py运行即可。preprocess.py和step1.py配合使用，包括处理标签、提取肺实质。

需要修改的地方：

1. prep_folder：处理后结果存放的路径
2. data_path：需要处理的dicom数据存放的地方。填入此的路径下有每一套LC015数据的文件夹，每个文件夹中存放该套CT的所有dicom文件。
3. alllabelfiles：这里存放的是一个excel表格，如果是其他格式，需要修改175行pandas读取的方式。我们的excel表格中，抬头是seriesuid、coordX、coordY、coordZ、length、id。我们只需要id和结节中心xyz坐标、结节长径。因此177行获取表格是1:6，filelist是获取所有ct名字，因此是最后一列，见179行的list(alllabel[:, **-1**])。同一套ct不同结节的id名字不需要修改，如LC01501001有3个结节，那这3个结节的数据分别各有一行，id这一列都为LC01501001。

目前的代码开了多线程，可以根据运行环境修改184行的Pool(8)这个数字，设置适合的线程数。建议不要太多，容易卡死。

处理后的文件夹中，同一套CT会得到如下5个文件夹。

1. xxx_clean.npy：提取了肺实质的图像，shape： [1, z, y, x]，第一个是指通道数，第二到四分别是z轴、y轴、x轴。
2. xxx_label.npy：处理过的和图像匹配的标签文件。单个结节的形式：[[z1, y1, x1, d1]]。多个结节形式：[[z1, y1, x1, d1], [z2, y2, x2, d2]]。没有结节的形式：[]
3. xxx_origin.npy：原点坐标。
4. xxx_spacing.npy：像素间距。
5. xxx_extendbox.npy：从原来dicom格式文件读出来的矩阵裁剪得到最后的clean.npy，如何裁剪，就是用extendbox。形式：[[z_start, z_end], [y_start, y_end], [x_start, x_end]]。

有用的就是clean和label两个文件。



#### 数据预处理2

SANet在处理PN9数据集的时候没有提取肺实质，因此我们提供了一个不提取肺实质的代码版本。

运行raw_preprocess.py，路径修改和数据预处理1的说明保持一致。

在这个处理方式里，进行了spacing归一化、HU值调整在[-1200, 600]区间，并使像素值线性缩放到[0, 255]的区间。



#### SANet的数据预处理

如果只是做测试的话，需要有三个文件。

1. txt文件：这个文件需要待测试数据的名字，一个名字一行，对应config.py里train_config的key值test_set_name所需要的txt文件。需要修改路径。
2. test_anno.csv文件：对应config.py里train_config的key值test_anno所需要的csv文件。这个文件夹里需要有pid、zmin、zmax、ymin、ymax、xmin、xmax的列。需要对应修改路径。
3. 用来计算指标的annotations，路径在test.py文件修改。这个csv文件需要有pid、center_x、center_y、center_z、diameter这几列，center_x、center_y、center_z是结节中心坐标，diameter是长径。

bbox_reader.py第90行需要修改一下名字。因为我们预处理之后得到的图像是xxxx_clean.npy。
