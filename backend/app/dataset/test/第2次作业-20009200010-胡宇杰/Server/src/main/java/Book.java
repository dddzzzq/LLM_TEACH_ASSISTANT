import java.io.Serializable;

//需要传输的类
public class Book implements Serializable {

    //无参数的构造方法
    public Book() {
        id=0;
        name="0";
    }
    //有参数的构造方法
    public Book(int id, String name){
        this.id=id;
        this.name=name;
    }


    public int GetId(){ return id;}
    public String GetName(){return name;}
    //重写toString函数，便于在返回书籍队列时使用
    public String toString() {
        return "Book{" +
                "name='" + name + '\'' +
                ", id=" + id +
                '}';
    }

    //私有变量
    private int id;
    private  String name;
    private static final long serialVersionUID = 1905122041950251207L;
}

