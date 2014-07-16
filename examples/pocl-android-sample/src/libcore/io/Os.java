package libcore.io;

public interface Os {
    public String getenv(String name);
    public void setenv(String name, String value, boolean overwrite);
}
