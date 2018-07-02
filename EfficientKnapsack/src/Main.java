import java.io.*;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Random;

public class Main {

    private static int n = 30;
    private static int[][] items;
    private static int max_weight = 0;

    public static int getN() {
        return n;
    }

    public static void setN(int n) {
        Main.n = n;
    }

    public static int[][] getItems() {
        return items;
    }

    public static void setItems(int[][] items) {
        Main.items = items;
    }

    public static int getMax_weight() {
        return max_weight;
    }

    public static void setMax_weight(int max_weight) {
        Main.max_weight = max_weight;
    }

    public Main() throws IOException {
        items = new int[n][3];
        buildItems();
    }

    private void buildItems() throws IOException {
//        Random random = new Random();
//        for (int i = 0; i < n; i++) {
//            int[] t = new int[3];
//            t[0] = i;
//            t[1] = random.nextInt(10) + 1;
//            t[2] = random.nextInt(10) + 1;
//            items[i] = t;
//        }


        FileInputStream fstream = new FileInputStream("5");
        DataInputStream in = new DataInputStream(fstream);
        BufferedReader br = new BufferedReader(new InputStreamReader(in));
        String data;
        max_weight = Integer.parseInt(br.readLine());
        System.out.println("max_weight: "+max_weight);
        int index = 0;
        while ((data = br.readLine()) != null)   {
            String[] tmp = data.split(" ");    //Split space
            int tmp_items[] = new int[3];
            tmp_items[0] = index;
            for (int i = 0; i < tmp.length; i++){
                    if (i % 2 == 0) {
//                        System.out.println("value: " + tmp[i]);
                        tmp_items[2] = Integer.parseInt(tmp[i]);
                    }
                    else{
//                    System.out.println("weight: " + tmp[i]);
                    tmp_items[1] = Integer.parseInt(tmp[i]);
                }
            }
            items[index] = tmp_items;
            index++;

        }
//        setN(index+1);

        for (int[] ar : items) {
            for (int el : ar)
                System.out.print(el + " ");
            System.out.println();
        }

        System.out.println();
    }


    public static void main(String[] args) throws IOException {
        Instant start = Instant.now();
        Main m = new Main();
        knapsack(getItems(), getMax_weight());
        Instant end = Instant.now();
        System.out.println("Execution time: "+Duration.between(start, end));
    }

    private static void knapsack(int[][] items, int capacity) {
        ArrayList<Integer> feasible = feasible(getN(), items, capacity);
        int knapsackDecimal = getOptimal(feasible, items, n);
        System.out.println("Binary characteristic vector: "+Integer.toBinaryString(knapsackDecimal));
        System.out.println("Optimal weight: "+getWeight(knapsackDecimal, items, n));
        System.out.println("Optimal value: "+getValue(knapsackDecimal, items, n));
    }

    private static int getOptimal(ArrayList<Integer> feasible, int[][] items, int n) {
        int bestValue = 0;
        int bestVector = 0;

        for (Integer vector : feasible){
            int value = getValue(vector, items, n);
            if (value <= bestValue)
                continue;
            bestVector = vector;
            bestValue = value;
        }
        return bestVector;
    }

    private static int getValue(Integer vector, int[][] items, int n) {
        int value = 0;
        for (int i = 0; i < n; i++){
            value += items[i][2] * isBitSet(vector, i);
        }
        return value;
    }

    private static ArrayList<Integer> feasible(int n, int[][] items, int capacity) {
        ArrayList<Integer> vectors = new ArrayList<Integer>();
        int max = (int) (Math.pow(2, n) - 1);
        int i = 0;
        while (i < max){
//            if (i % 1000000 == 0)
//                System.out.println(i);
            if (getWeight(i, items, n) <= capacity)
                vectors.add(i);
            i++;
        }
        return vectors;
    }

    private static int getWeight(int vector, int[][] items, int n) {
        int weight = 0;
        for (int i = 0; i < n; i++){
            weight += items[i][1] * isBitSet(vector, i);
        }
        return weight;
    }

    private static int isBitSet(int vector, int i) {
        return (vector >> i) & 1;
    }
}
