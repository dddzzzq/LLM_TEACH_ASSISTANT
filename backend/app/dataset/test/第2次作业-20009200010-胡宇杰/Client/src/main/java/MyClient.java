import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.util.ArrayList;
import java.util.Scanner;

public class MyClient {
    public static Scanner scanner = new Scanner(System.in);

    //客户端程序入口
    public static void main(String[] args) {

        try {
            String name = "BookSystem";
            String serverIP = "127.0.0.1";  // 注册中心的IP地址
            int serverPort = 1099;// 注册中心的端口号
            //获取注册中心引用
            Registry registry = LocateRegistry.getRegistry(serverIP, serverPort);
            Operations operation = (Operations) registry.lookup(name);

            //初始化
            System.out.println("初始化中...");
            init(operation);
            System.out.println("初始化成功！\n");

            //功能实现
            System.out.println("这里是一个远程图书管理系统，下面是我的功能说明：");
            System.out.println("输入1可实现：书籍添加");
            System.out.println("输入2可实现：根据书籍编号查询书籍");
            System.out.println("输入3可实现：根据书籍名称查询书籍");
            System.out.println("输入4可实现：根据书籍编号删除对应书籍");
            System.out.println("输入0可实现：退出程序");

            //循环实现相应功能
            for(;;){
                System.out.println("请输入：");

                //获取输入，根据输入判断需要调用的函数
                int num=scanner.nextInt();
                if(num==0){

                    //在服务器端列出所有的图书，便于判断程序是否运行正确
                    operation.ListAll();
                    return;
                }
                else if(num==1){
                    function1(operation);
                }
                else if(num==2){
                    function2(operation);
                }
                else if(num==3){
                    function3(operation);
                }
                else if(num==4){
                    function4(operation);
                }
                else{
                    System.out.println("请输入正确的功能编号！");
                }
            }

            //功能测试
            /*ArrayList booklist=operation.queryByName("数据");
            System.out.println(booklist.toString());
            operation.queryByID(6);
            operation.queryByID(2);
            DELETE(operation,6);
            DELETE(operation,3);*/


        } catch (Exception e) {
            System.err.println("??? exception:");
            e.printStackTrace();
        }
    }


    //具体功能实现
    public static void ADD(Operations operation,Book book) throws RemoteException {
        boolean result=operation.add(book);
        System.out.print("The book 《"+book.GetName()+"》 ");
        if(result)  System.out.println("添加成功！");
        else System.out.println("添加失败！");
    }
    public static void QUERYBYID(Operations operation,int id) throws RemoteException {
        Book book=operation.queryByID(id);
        System.out.println("The book is 《"+book.GetName()+"》 ");
    }
    public static void QUERYBYNAME(Operations operation,String name) throws RemoteException {
        ArrayList booklist=operation.queryByName(name);
        System.out.println(booklist.toString());
    }
    public static void DELETE(Operations operation,int bookID) throws RemoteException {
        boolean result=operation.delete(bookID);
        System.out.print("Result : ");
        if(result)  System.out.println("删除成功！");
        else System.out.println("删除失败！");
    }

    public static void function1(Operations operation) throws RemoteException{
        String name=scanner.nextLine();
        System.out.println("请输入想要添加的书籍的名称：");
        name=scanner.nextLine();
        System.out.println("请输入想要添加的书籍的编号：");
        int id=scanner.nextInt();
        Book book=new Book(id,name);
        ADD(operation,book);
    }
    public static void function2(Operations operation) throws RemoteException{
        System.out.println("请输入想要查询的书籍的编号：");
        int id=scanner.nextInt();
        QUERYBYID(operation,id);
    }
    public static void function3(Operations operation) throws RemoteException{
        String name=scanner.nextLine();
        System.out.println("请输入想要查询的书籍的名称：");
        name=scanner.nextLine();
        QUERYBYNAME(operation,name);
    }
    public static void function4(Operations operation) throws RemoteException{
        System.out.println("请输入想要查删除的书籍的编号：");
        int id=scanner.nextInt();
        DELETE(operation,id);
    }
    public static void init(Operations operation) throws RemoteException{
        Book book1=new Book(1,"数据结构");
        Book book2=new Book(2,"算法");
        Book book3=new Book(3,"数据库");
        Book book4=new Book(4,"分布式计算");
        Book book5=new Book(5,"母猪的产后护理");
        ADD(operation, book1);
        ADD(operation, book2);
        ADD(operation, book3);
        ADD(operation, book4);
        ADD(operation, book5);
    }
}
