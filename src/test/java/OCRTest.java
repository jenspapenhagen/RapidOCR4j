import io.github.hzkitty.RapidOCR;
import io.github.hzkitty.entity.OcrResult;
import io.github.hzkitty.entity.ParamConfig;
import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class OCRTest {

//    static {
//        nu.pattern.OpenCV.loadShared();
//    }

    @Test
    public void testPath() throws Exception {
        final RapidOCR rapidOCR = RapidOCR.create();
        final File file = new File("src/test/resources/text_01.png");
        final String imgContent = file.getAbsolutePath();
        final OcrResult ocrResult = rapidOCR.run(imgContent);
        assertFalse(ocrResult.recRes().isEmpty());
        assertTrue(ocrResult.toString().startsWith("OcrResult{strRes='【Tabula-Java的优点与不足】\n" +
                "Tabula-Java的优点有：\n" +
                "1.开源免费：Tabula-Java是开源的，用户可以免费使用。\n" +
                "2.功能强大：Tabula-Java支持多种表格格式，同时可以从PDF文件中提取图片。\n" +
                "3.易于集成：Tabula-Java基于Java编写，可以方便地集成到其他Java应用程序中。\n" +
                "Tabula-Java的不足有：\n" +
                "1.仅支持Java：Tabula-Java仅支持Java语言，对于其他编程语言的用户来说不太友好。"));
    }

    @Test
    public void testBufferedImage() throws Exception {
        final RapidOCR rapidOCR = RapidOCR.create();
        final File file = new File("src/test/resources/text_01.png");
        final BufferedImage imgContent = ImageIO.read(file);

        final ParamConfig paramConfig = new ParamConfig();
        //paramConfig.returnWordBox(true); default is true
        final OcrResult ocrResult = rapidOCR.run(imgContent, paramConfig);
        assertFalse(ocrResult.recRes().isEmpty());
        assertTrue(ocrResult.toString().startsWith("OcrResult{strRes='【Tabula-Java的优点与不足】\n" +
                "Tabula-Java的优点有：\n" +
                "1.开源免费：Tabula-Java是开源的，用户可以免费使用。\n" +
                "2.功能强大：Tabula-Java支持多种表格格式，同时可以从PDF文件中提取图片。\n" +
                "3.易于集成：Tabula-Java基于Java编写，可以方便地集成到其他Java应用程序中。\n" +
                "Tabula-Java的不足有：\n" +
                "1.仅支持Java：Tabula-Java仅支持Java语言，对于其他编程语言的用户来说不太友好。"));
    }

    @Test
    public void testByte() throws Exception {
        final RapidOCR rapidOCR = RapidOCR.create();
        final File file = new File("src/test/resources/text_01.png");
        byte[] imgContent = Files.readAllBytes(file.toPath());
        final OcrResult ocrResult = rapidOCR.run(imgContent);
        assertFalse(ocrResult.recRes().isEmpty());
        assertTrue(ocrResult.toString().startsWith("OcrResult{strRes='【Tabula-Java的优点与不足】\n" +
                "Tabula-Java的优点有：\n" +
                "1.开源免费：Tabula-Java是开源的，用户可以免费使用。\n" +
                "2.功能强大：Tabula-Java支持多种表格格式，同时可以从PDF文件中提取图片。\n" +
                "3.易于集成：Tabula-Java基于Java编写，可以方便地集成到其他Java应用程序中。\n" +
                "Tabula-Java的不足有：\n" +
                "1.仅支持Java：Tabula-Java仅支持Java语言，对于其他编程语言的用户来说不太友好。"));
    }

    @Test
    public void testMat() throws Exception {
        final RapidOCR rapidOCR = RapidOCR.create();
        final File file = new File("src/test/resources/text_01.png");
        final Mat imgContent = Imgcodecs.imread(file.getAbsolutePath());
        final OcrResult ocrResult = rapidOCR.run(imgContent);
        assertFalse(ocrResult.recRes().isEmpty());
        assertTrue(ocrResult.toString().startsWith("OcrResult{strRes='【Tabula-Java的优点与不足】\n" +
                "Tabula-Java的优点有：\n" +
                "1.开源免费：Tabula-Java是开源的，用户可以免费使用。\n" +
                "2.功能强大：Tabula-Java支持多种表格格式，同时可以从PDF文件中提取图片。\n" +
                "3.易于集成：Tabula-Java基于Java编写，可以方便地集成到其他Java应用程序中。\n" +
                "Tabula-Java的不足有：\n" +
                "1.仅支持Java：Tabula-Java仅支持Java语言，对于其他编程语言的用户来说不太友好。"));
    }

}
