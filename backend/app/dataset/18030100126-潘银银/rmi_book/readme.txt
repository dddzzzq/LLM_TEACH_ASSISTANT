
book.txt 存放了书籍信息；
.java 文件文件为源文件；


使用方法：
开3个cmd窗口，都需要切换到改文件夹的路径下运行

命令顺序：
javac *.java
rmic ComputingServiceImpl   
java RMIServer
java RMIClient


//cmd1
---------------------------------------------------------
实例演示结果：
C:\Users\nsus\Desktop\rmi_book>javac *.java

C:\Users\nsus\Desktop\rmi_book>rmic ComputingServiceImpl
警告: 为 JRMP 生成和使用骨架及静态存根
已过时。骨架不再必要, 而静态存根
已由动态生成的存根取代。建议用户
不再使用rmic来生成骨架和静态存根。
请参阅 java.rmi.server.UnicastRemoteObject 的文档。
---------------------------------------------------------


//cmd2
---------------------------------------------------------
实例演示结果：
C:\Users\nsus\Desktop\rmi_book>java RMIServer
ComputingService of book manage system is online.
---------------------------------------------------------


//cmd3
---------------------------------------------------------
实例演示结果：
C:\Users\nsus\Desktop\rmi_book>java RMIClient
----Book Manager System----
0.exit
1.add book
2.query Book by ID
3.query Book by keyword
4.delete book
5.show all books
---------------------------
5
-------all books-------
id:2 name:English
id:4 name:zr
id:1 name:math
id:3 name:dance
id:5 name:java

----Book Manager System----
0.exit
1.add book
2.query Book by ID
3.query Book by keyword
4.delete book
5.show all books
---------------------------
4
Please input the keyword of the Book that you want to delete:
4
-----delete successful-----
----Book Manager System----
0.exit
1.add book
2.query Book by ID
3.query Book by keyword
4.delete book
5.show all books
---------------------------
5
-------all books-------
id:2 name:English
id:1 name:math
id:3 name:dance
id:5 name:java

----Book Manager System----
0.exit
1.add book
2.query Book by ID
3.query Book by keyword
4.delete book
5.show all books
---------------------------
2
Please input the id  of the Book that you want to find:
3
-------results-------
ID: 3  name:dance

----Book Manager System----
0.exit
1.add book
2.query Book by ID
3.query Book by keyword
4.delete book
5.show all books
---------------------------
3
Please input the keyword of the Book that you want to find:
h
-------results-------
id:2 name:English
id:1 name:math

----Book Manager System----
0.exit
1.add book
2.query Book by ID
3.query Book by keyword
4.delete book
5.show all books
---------------------------
---------------------------------------------------------


















