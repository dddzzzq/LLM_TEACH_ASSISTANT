import java.util.*;
import java.io.*;
import java.rmi.*;
import java.net.*;
import java.rmi.registry.LocateRegistry;


public class RMIServer {
    public static void main(String[] args) throws RemoteException, AlreadyBoundException, MalformedURLException {
        
        ComputingService computingServant = new ComputingServiceImpl();
        LocateRegistry.createRegistry(8889);
        Naming.bind("rmi://localhost:8889/ComputingService",computingServant);
        System.out.println("ComputingService of book manage system is online.");
    }
}
