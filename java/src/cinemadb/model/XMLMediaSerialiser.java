package cinemadb.model;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import cinemadb.view.Constants;
import cinemadb.view.enums.Genre;
import cinemadb.view.enums.MediaType;
import javax.xml.namespace.QName;
import javax.xml.stream.FactoryConfigurationError;
import javax.xml.stream.XMLEventReader;
import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLOutputFactory;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamWriter;
import javax.xml.stream.events.Attribute;
import javax.xml.stream.events.StartElement;
import javax.xml.stream.events.XMLEvent;

public class XMLMediaSerialiser {

	private static XMLMediaSerialiser instance;

	private XMLMediaSerialiser() {
	}

	public synchronized static XMLMediaSerialiser getInstance() {
		if (instance == null) {
			instance = new XMLMediaSerialiser();
		}
		return instance;
	}

	public List<Media> readAll() throws FileNotFoundException, XMLStreamException, FactoryConfigurationError {
		XMLEventReader reader = XMLInputFactory.newInstance()
			.createXMLEventReader(new FileInputStream(Constants.MEDIA_FILE_PATH));

		List<Media> mediaList = new ArrayList<>();
		Media media = null;

		while (reader.hasNext()) {
			XMLEvent nextEvent = reader.nextEvent();

			if (nextEvent.isStartElement()) {
				StartElement startElement = nextEvent.asStartElement();

				switch (startElement.getName().getLocalPart()) {
					case "media":
						media = new Media();
						Attribute idAtt = startElement.getAttributeByName(new QName("id"));
						media.setId(Integer.parseInt(idAtt.getValue()));
						break;
					case "title":
						Attribute titleAtt = startElement.getAttributeByName(new QName("value"));
						media.setTitle(titleAtt.getValue());
						break;
					case "mediaType":
						Attribute typeAtt = startElement.getAttributeByName(new QName("value"));
						media.setMediaType(MediaType.getFromStr(typeAtt.getValue()));
						break;
					case "year":
						Attribute yearAtt = startElement.getAttributeByName(new QName("value"));
						media.setYear(Integer.parseInt(yearAtt.getValue()));
						break;
					case "genre":
						Attribute genreAtt = startElement.getAttributeByName(new QName("value"));
						media.getGenres().add(Genre.getFromStr(genreAtt.getValue()));
						break;
					case "director":
						Attribute directorAtt = startElement.getAttributeByName(new QName("name"));
						media.getDirectors().add(directorAtt.getValue());
						break;
					case "actor":
						Attribute actorAtt = startElement.getAttributeByName(new QName("name"));
						media.getActors().add(actorAtt.getValue());
						break;
				}
			} else if (nextEvent.isEndElement() && nextEvent.asEndElement().getName().getLocalPart().equals("media")) {
				// If reached </media> tag
				mediaList.add(media);
			}
		}
		reader.close();
		return mediaList;
	}

	public void write(List<Media> mediaList) throws XMLStreamException, FactoryConfigurationError, IOException {
		XMLStreamWriter writer = XMLOutputFactory.newInstance()
			.createXMLStreamWriter(new FileWriter(Constants.MEDIA_FILE_PATH));

		writer.writeStartDocument();
		writer.writeStartElement("mediaList");

		for (Media media : mediaList) {
			writeMediaElement(writer, media);
		}

		writer.writeEndElement(); // Write </mediaList> tag
		writer.writeEndDocument();
		writer.flush();
		writer.close();
	}

	private void writeMediaElement(XMLStreamWriter writer, Media media) throws XMLStreamException {
		writer.writeStartElement("media");
		writer.writeAttribute("id", Integer.toString(media.getId()));

		writer.writeStartElement("title");
		writer.writeAttribute("value", media.getTitle());
		writer.writeEndElement(); // Write </title> tag

		writer.writeStartElement("mediaType");
		writer.writeAttribute("value", media.getMediaType().toString());
		writer.writeEndElement(); // Write </mediaType> tag

		writer.writeStartElement("year");
		writer.writeAttribute("value", Integer.toString(media.getYear()));
		writer.writeEndElement(); // Write </year> tag

		writer.writeStartElement("genres");
		for (Genre genre : media.getGenres()) {
			writer.writeStartElement("genre");
			writer.writeAttribute("value", genre.toString());
			writer.writeEndElement(); // Write </genre> tag
		}
		writer.writeEndElement(); // Write </genres> tag

		writer.writeStartElement("directors");
		for (String name : media.getDirectors()) {
			writer.writeStartElement("director");
			writer.writeAttribute("name", name);
			writer.writeEndElement(); // Write </director> tag
		}
		writer.writeEndElement(); // Write </directors> tag

		writer.writeStartElement("actors");
		for (String name : media.getActors()) {
			writer.writeStartElement("actor");
			writer.writeAttribute("name", name);
			writer.writeEndElement(); // Write </actor> tag
		}
		writer.writeEndElement(); // Write </actors> tag

		writer.writeEndElement(); // Write </media> tag
	}
}
