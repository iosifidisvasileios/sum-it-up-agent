from sum_it_up_agent.communicator import CommunicatorFactory, CommunicatorConfig, ChannelType, CommunicationRequest

comm = CommunicatorFactory.create(
    CommunicatorConfig(channel=ChannelType.EMAIL)
)


req = CommunicationRequest(
    recipient="billiosifidis@gmail.com",
    subject="Meeting summary: planning / coordination meeting",
    path='/home/vios/PycharmProjects/sum-it-up-agent/examples/ollama_summaries/Product_Marketing_transcription_summary.json',
)

comm.send(req)