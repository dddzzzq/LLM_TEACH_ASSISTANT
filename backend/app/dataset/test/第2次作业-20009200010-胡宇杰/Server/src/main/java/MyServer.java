import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.rmi.server.UnicastRemoteObject;

public class MyServer {
    public static void main(String[] args) throws Exception {

        try {
            String name = "BookSystem";
            Operations engine = new Operationslmpl();
            // 生成包裹engine对象的容器对象，即skeleton对象
            Operations skeleton = (Operations) UnicastRemoteObject.exportObject(engine, 0);
            // 获取注册中心的引用，示例中，注册中心运行在本地计算机上。
            Registry registry = LocateRegistry.createRegistry( 1099);
            System.out.println("Registering BookSystem Object");
            registry.rebind(name, skeleton);
        } catch (Exception e) {
            System.err.println("Exception:" + e);
            e.printStackTrace();
        }
    }
}
